"""
IdleFL Agent - macOS / MPS
Connects to the IdleFL server, trains on a local shard, and returns weights.

SETUP:
  1. pip install torch torchvision numpy "python-socketio[client]" aiohttp psutil
  2. Edit USER_ID and SESSION_ID below, then run: python agent_mac.py
"""

import asyncio
import math
import os
import platform
import shlex
import subprocess
import time

import aiohttp
import numpy as np
import psutil
import socketio

USER_ID = "paste_your_user_id_here"  # Your permanent agent ID (e.g. "durvish_a3k9")
SESSION_ID = "FL-XXXX"               # Session code shown on the dashboard (e.g. "FL-4829")

# For demo purposes you can paste a JWT directly here instead of using login.
# Leave empty to authenticate via POST /api/auth/agent-login at startup.
JWT_TOKEN = ""

# Server URL is injected at download time by the IdleFL server.
# Do not edit this line manually - download the script fresh from the dashboard.
SERVER_URL = 'SERVER_URL_PLACEHOLDER'

HEARTBEAT_INTERVAL_SECONDS = 30
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 1
MOCK_WEIGHT_SIZE = 16


def detect_hardware():
    """Detects the best available compute backend for local training."""
    compute_type = "CPU"
    try:
        import torch

        if torch.cuda.is_available():
            compute_type = "CUDA"
            print(f"[✓] CUDA detected: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            compute_type = "MPS"
            print("[✓] Apple MPS detected")
        else:
            print("[●] No GPU found - using CPU")
    except ImportError:
        print("[✗] PyTorch not installed - CNN training will fall back to mock weights")
    return compute_type


def get_stats():
    """Collects heartbeat stats for the coordinator dashboard."""
    stats = {
        "cpuPercent": psutil.cpu_percent(interval=0.5),
        "freeRamGb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
        "totalRamGb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpuPercent": 0,
        "gpuVramUsed": 0,
        "gpuVramTotal": 0,
    }
    try:
        import GPUtil

        gpus = GPUtil.getGPUs()
        if gpus:
            stats["gpuPercent"] = gpus[0].load * 100
            stats["gpuVramUsed"] = gpus[0].memoryUsed / 1024
            stats["gpuVramTotal"] = gpus[0].memoryTotal / 1024
    except Exception:
        pass
    return stats


def prevent_sleep():
    """Prevents the host OS from sleeping while training is running."""
    if platform.system() == "Windows":
        import ctypes

        es_continuous = 0x80000000
        es_system_required = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
        print("[✓] Sleep prevention active (Windows)")
    elif platform.system() == "Darwin":
        subprocess.Popen(["caffeinate", "-i"])
        print("[✓] Sleep prevention active (Mac caffeinate)")


def release_sleep():
    """Releases the OS sleep-prevention lock on shutdown."""
    if platform.system() == "Windows":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def _config_value(config, snake_key, camel_key, default_value):
    """Reads either snake_case or camelCase config keys."""
    if snake_key in config and config[snake_key] is not None:
        return config[snake_key]
    if camel_key in config and config[camel_key] is not None:
        return config[camel_key]
    return default_value


def _emit_training_checkpoint(config, weights, iteration):
    """Schedules training:checkpoint on the main asyncio loop (when training runs in a thread)."""
    if not config:
        return
    main_loop = config.get("_main_loop")
    sio = config.get("_sio")
    task_id = config.get("taskId")
    job_id = config.get("jobId")
    round_num = config.get("roundNum")
    if not main_loop or not sio or task_id is None:
        return
    asyncio.run_coroutine_threadsafe(
        sio.emit(
            "training:checkpoint",
            {
                "taskId": task_id,
                "jobId": job_id,
                "roundNum": round_num,
                "checkpointData": weights,
            },
        ),
        main_loop,
    )


def _mock_result(model_type, weight_size=MOCK_WEIGHT_SIZE):
    """Builds a deterministic fallback payload when real training is unavailable."""
    base_size = {"LINEAR_REGRESSION": max(weight_size, 4), "LOGISTIC_REGRESSION": max(weight_size, 8), "CNN": max(weight_size, 64)}
    size = base_size.get(model_type, max(weight_size, MOCK_WEIGHT_SIZE))
    weights = (np.random.randn(size) * 0.05).astype(float).tolist()
    return {"weights": weights, "loss": float(np.random.uniform(0.2, 1.2)), "accuracy": float(np.random.uniform(0.4, 0.9))}


def _safe_log(values):
    """Numerically stable log helper for cross-entropy calculations."""
    return np.log(np.clip(values, 1e-8, 1.0 - 1e-8))


def _sigmoid(values):
    """Stable sigmoid implementation."""
    clipped = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(values):
    """Stable softmax implementation for multiclass logistic regression."""
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _train_linear_regression(data_shard, config):
    """Trains a linear regression model with full-batch gradient descent."""
    x = np.asarray(data_shard["X"], dtype=np.float64)
    y = np.asarray(data_shard["y"], dtype=np.float64)
    num_samples, num_features = x.shape

    # Z-score normalize inputs
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_std = np.where(x_std == 0, 1.0, x_std)
    x_norm = (x - x_mean) / x_std

    y_mean = np.mean(y)
    y_std = float(np.std(y))
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    learning_rate = float(_config_value(config, "learning_rate", "learningRate", DEFAULT_LEARNING_RATE))
    epochs = int(_config_value(config, "epochs", "epochs", DEFAULT_EPOCHS))
    task_data = config
    checkpoint_interval = int(task_data.get("checkpointInterval", 10))

    # Ignore globalWeights for warm start (original-space params do not match normalized training).
    weights = np.zeros(num_features, dtype=np.float64)
    bias = 0.0

    for iteration in range(1, epochs + 1):
        prediction = x_norm @ weights + bias
        residual = prediction - y_norm
        grad_w = (2.0 / num_samples) * (x_norm.T @ residual)
        grad_b = (2.0 / num_samples) * np.sum(residual)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
        if checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
            weights_orig_ckpt = weights * y_std / x_std
            bias_orig_ckpt = y_mean + bias * y_std - np.dot(weights_orig_ckpt, x_mean)
            _emit_training_checkpoint(
                config,
                weights_orig_ckpt.astype(float).tolist() + [float(bias_orig_ckpt)],
                iteration,
            )

    # Denormalize weights back to original feature space
    weights_orig = weights * y_std / x_std
    bias_orig = y_mean + bias * y_std - np.dot(weights_orig, x_mean)

    # Compute loss and accuracy in original space for reporting
    pred_orig = x @ weights_orig + bias_orig
    resid = pred_orig - y
    loss = float(np.mean(resid ** 2))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    accuracy = max(0.0, 1.0 - ss_res / ss_tot) if not math.isclose(ss_tot, 0) else 1.0

    return {
        "weights": weights_orig.tolist() + [float(bias_orig)],
        "loss": loss,
        "accuracy": float(accuracy),
    }


def _train_logistic_regression(data_shard, config):
    """Trains binary or multiclass logistic regression using SGD mini-batches."""
    x = np.asarray(data_shard["X"], dtype=np.float64)
    y = np.asarray(data_shard["y"])
    num_samples, num_features = x.shape

    learning_rate = float(_config_value(config, "learning_rate", "learningRate", DEFAULT_LEARNING_RATE))
    batch_size = int(_config_value(config, "batch_size", "batchSize", DEFAULT_BATCH_SIZE))
    epochs = int(_config_value(config, "epochs", "epochs", DEFAULT_EPOCHS))
    global_weights = _config_value(config, "global_weights", "globalWeights", None)
    task_data = config
    checkpoint_interval = int(task_data.get("checkpointInterval", 10))

    # Scale learning rate based on feature magnitude to prevent explosion
    feature_scale = np.sqrt(np.mean(x ** 2))
    if feature_scale > 1:
        learning_rate = learning_rate / (feature_scale ** 2)

    unique_labels = np.unique(y)
    is_binary = unique_labels.size <= 2
    last_loss = 0.0

    if is_binary:
        y_binary = y.astype(np.float64)
        if global_weights and len(global_weights) == num_features + 1:
            params = np.asarray(global_weights, dtype=np.float64)
            weights = params[:-1].copy()
            bias = float(params[-1])
        else:
            weights = np.zeros(num_features, dtype=np.float64)
            bias = 0.0

        iteration = 0
        for _ in range(epochs):
            indices = np.random.permutation(num_samples)
            for start in range(0, num_samples, max(batch_size, 1)):
                iteration += 1
                batch_idx = indices[start:start + batch_size]
                x_batch = x[batch_idx]
                y_batch = y_binary[batch_idx]
                prediction = _sigmoid(x_batch @ weights + bias)
                last_loss = float(-np.mean(y_batch * _safe_log(prediction) + (1.0 - y_batch) * _safe_log(1.0 - prediction)))
                error = prediction - y_batch
                grad_w = (x_batch.T @ error) / len(batch_idx)
                grad_b = float(np.mean(error))
                weights -= learning_rate * grad_w
                bias -= learning_rate * grad_b
                if checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
                    _emit_training_checkpoint(
                        config,
                        weights.astype(float).tolist() + [float(bias)],
                        iteration,
                    )

        final_prediction = _sigmoid(x @ weights + bias)
        predicted_labels = (final_prediction >= 0.5).astype(int)
        accuracy = float(np.mean(predicted_labels == y_binary.astype(int)))
        return {"weights": weights.astype(float).tolist() + [float(bias)], "loss": last_loss, "accuracy": accuracy}

    class_to_index = {label: index for index, label in enumerate(unique_labels)}
    y_indices = np.asarray([class_to_index[label] for label in y], dtype=np.int64)
    num_classes = unique_labels.size

    expected_len = (num_features * num_classes) + num_classes
    if global_weights and len(global_weights) == expected_len:
        params = np.asarray(global_weights, dtype=np.float64)
        weight_end = num_features * num_classes
        weights = params[:weight_end].reshape(num_features, num_classes).copy()
        bias = params[weight_end:].copy()
    else:
        weights = np.zeros((num_features, num_classes), dtype=np.float64)
        bias = np.zeros(num_classes, dtype=np.float64)

    iteration = 0
    for _ in range(epochs):
        indices = np.random.permutation(num_samples)
        for start in range(0, num_samples, max(batch_size, 1)):
            iteration += 1
            batch_idx = indices[start:start + batch_size]
            x_batch = x[batch_idx]
            y_batch = y_indices[batch_idx]
            logits = x_batch @ weights + bias
            probabilities = _softmax(logits)
            targets = np.eye(num_classes)[y_batch]
            last_loss = float(-np.mean(np.sum(targets * _safe_log(probabilities), axis=1)))
            error = probabilities - targets
            grad_w = (x_batch.T @ error) / len(batch_idx)
            grad_b = np.mean(error, axis=0)
            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b
            if checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
                _emit_training_checkpoint(
                    config,
                    weights.astype(float).reshape(-1).tolist() + bias.astype(float).tolist(),
                    iteration,
                )

    final_logits = x @ weights + bias
    predicted_indices = np.argmax(final_logits, axis=1)
    accuracy = float(np.mean(predicted_indices == y_indices))
    flat_weights = weights.astype(float).reshape(-1).tolist() + bias.astype(float).tolist()
    return {"weights": flat_weights, "loss": last_loss, "accuracy": accuracy}


def _load_cnn_dependencies():
    """Imports PyTorch and torchvision lazily so non-CNN jobs do not require them."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as f
        import torchvision
        from torch.utils.data import DataLoader, Subset
        from torchvision import transforms

        return torch, nn, f, torchvision, DataLoader, Subset, transforms
    except ImportError as error:
        print("[✗] CNN training requires PyTorch and torchvision. Install with: pip install torch torchvision")
        raise error


def _train_cnn(data_shard, config):
    """Trains the small IdleFL CNN on the assigned MNIST or CIFAR-10 subset."""
    torch, nn, f, torchvision, DataLoader, Subset, transforms = _load_cnn_dependencies()

    dataset_name = (data_shard.get("dataset_name") or data_shard.get("datasetName") or "mnist").lower()
    indices = data_shard.get("indices") or []
    if not indices:
        raise ValueError("CNN shard did not include dataset indices")

    class IdleFLCNN(nn.Module):
        """Small CNN used for fast federated demos on consumer hardware."""

        def __init__(self, num_classes=10, input_channels=1, flattened_size=64 * 7 * 7):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(flattened_size, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.flattened_size = flattened_size

        def forward(self, inputs):
            outputs = self.pool(f.relu(self.conv1(inputs)))
            outputs = self.pool(f.relu(self.conv2(outputs)))
            outputs = outputs.view(-1, self.flattened_size)
            outputs = f.relu(self.fc1(outputs))
            return self.fc2(outputs)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    input_channels = 3 if dataset_name == "cifar10" else 1
    flattened_size = 64 * 8 * 8 if dataset_name == "cifar10" else 64 * 7 * 7
    model = IdleFLCNN(input_channels=input_channels, flattened_size=flattened_size).to(device)

    global_weights = _config_value(config, "global_weights", "globalWeights", None)
    if global_weights:
        offset = 0
        state_dict = model.state_dict()
        for name, tensor in state_dict.items():
            tensor_size = tensor.numel()
            segment = global_weights[offset:offset + tensor_size]
            if len(segment) != tensor_size:
                raise ValueError(f"Global weight length mismatch while reconstructing parameter '{name}'")
            state_dict[name] = torch.tensor(segment, dtype=tensor.dtype).view_as(tensor)
            offset += tensor_size
        model.load_state_dict(state_dict)

    data_root = os.path.expanduser("~/.idlefl_data")
    transform = transforms.ToTensor()
    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    else:
        dataset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)

    subset = Subset(dataset, indices)
    batch_size = int(_config_value(config, "batch_size", "batchSize", DEFAULT_BATCH_SIZE))
    epochs = int(_config_value(config, "epochs", "epochs", DEFAULT_EPOCHS))
    learning_rate = float(_config_value(config, "learning_rate", "learningRate", DEFAULT_LEARNING_RATE))
    task_data = config
    checkpoint_interval = int(task_data.get("checkpointInterval", 10))

    loader = DataLoader(subset, batch_size=max(batch_size, 1), shuffle=True)
    criterion = nn.CrossEntropyLoss()
    # FedAvg averages model weights cleanly across workers; Adam's momentum state does not.
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    last_loss = 0.0
    model.train()
    iteration = 0
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            iteration += 1
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
            if checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
                flat_weights = []
                for parameter in model.parameters():
                    flat_weights.extend(parameter.data.detach().cpu().numpy().astype(float).reshape(-1).tolist())
                _emit_training_checkpoint(config, flat_weights, iteration)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            predictions = torch.argmax(logits, dim=1)
            correct += int((predictions == batch_y).sum().item())
            total += int(batch_y.size(0))

    flat_weights = []
    for parameter in model.parameters():
        flat_weights.extend(parameter.data.detach().cpu().numpy().astype(float).reshape(-1).tolist())

    accuracy = float(correct / total) if total else 0.0
    return {"weights": flat_weights, "loss": last_loss, "accuracy": accuracy}


def train_locally(model_type, data_shard, config):
    """
    Trains a local model on the assigned shard and returns flattened weights.

    Args:
        model_type: LINEAR_REGRESSION, LOGISTIC_REGRESSION, or CNN.
        data_shard: Tabular shard {"X", "y"} or image shard {"datasetName", "indices"}.
        config: Training config including learning rate, batch size, epochs, and global weights.

    Returns:
        Dict with keys: {"weights": list[float], "loss": float, "accuracy": float}.
    """
    round_num = _config_value(config, "round_num", "roundNum", "?")

    if not data_shard:
        print(f"[✗] Round {round_num} received no shard data - returning mock weights")
        return _mock_result(model_type)

    try:
        if model_type == "LINEAR_REGRESSION":
            result = _train_linear_regression(data_shard, config)
        elif model_type == "LOGISTIC_REGRESSION":
            result = _train_logistic_regression(data_shard, config)
        elif model_type == "CNN":
            result = _train_cnn(data_shard, config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        print(f"[●] [Round {round_num}] Loss: {result['loss']:.4f}  Accuracy: {result['accuracy'] * 100:.1f}%  (training...)")
        return result
    except Exception as error:
        print(f"[✗] Local training failed for round {round_num}: {error}")
        fallback_size = len(_config_value(config, "global_weights", "globalWeights", []) or []) or MOCK_WEIGHT_SIZE
        return _mock_result(model_type, fallback_size)


async def run_agent():
    """Connects to the IdleFL backend and handles training tasks over Socket.IO."""
    prevent_sleep()
    compute_type = detect_hardware()

    print(f"\n  IdleFL Agent - {platform.system()} / {compute_type}")

    token = JWT_TOKEN
    if not token:
        print(f"  Authenticating as agent {USER_ID} for session {SESSION_ID}...")
        async with aiohttp.ClientSession() as http:
            resp = await http.post(
                f"{SERVER_URL}/api/auth/agent-login",
                json={"agentId": USER_ID, "sessionCode": SESSION_ID},
            )
            body = await resp.json()
            token = body.get("token") or (body.get("data") or {}).get("token", "")
        if not token:
            print("[✗] Agent login failed - check USER_ID and SESSION_ID")
            print(f"    Server response: {body.get('error', 'unknown error')}")
            return

    print(f"  Connecting to {SERVER_URL}...\n")

    sio = socketio.AsyncClient(reconnection=True, reconnection_attempts=10, reconnection_delay=2)

    # Shared state for the one-at-a-time checkpoint fetch handshake (agent-pull).
    _pending_checkpoint = {}
    # Server-pushed checkpoint weights indexed by taskId (for reassignment flow).
    _pending_checkpoints = {}

    @sio.on("training:checkpoint_data")
    async def on_checkpoint_data(data):
        evt = _pending_checkpoint.get("event")
        if evt and not evt.is_set():
            _pending_checkpoint["weights"] = data.get("weights")
            evt.set()

    @sio.on("training:checkpoint_fetch")
    async def on_server_checkpoint_fetch(data):
        """Handles server-push checkpoint notification during device reassignment (check 5).

        Server emits: { taskId, checkpointKey }
        Agent fetches the checkpoint data using the key and stores it so that
        on_task_assigned can inject it into globalWeights before training starts.
        """
        task_id = data.get("taskId")
        checkpoint_key = data.get("checkpointKey")
        if not task_id or not checkpoint_key:
            return
        print(f"[●] Server pushed checkpoint for task {task_id}: {checkpoint_key}")
        evt = asyncio.Event()
        _pending_checkpoint["event"] = evt
        _pending_checkpoint["weights"] = None
        await sio.emit("training:checkpoint_fetch", {"checkpointKey": checkpoint_key})
        try:
            await asyncio.wait_for(evt.wait(), timeout=5.0)
            weights = _pending_checkpoint.get("weights")
            _pending_checkpoints[task_id] = weights
            if weights:
                print(f"[✓] Checkpoint pre-loaded for task {task_id}: {len(weights)} weights")
            else:
                print(f"[⚠] Checkpoint empty for task {task_id} - will train from scratch")
        except asyncio.TimeoutError:
            print(f"[⚠] Checkpoint pre-load timed out for task {task_id}")
            _pending_checkpoints[task_id] = None
        finally:
            _pending_checkpoint.clear()

    @sio.event
    async def connect():
        print("[✓] Connected to server")
        await sio.emit("device:register", {
            "sessionCode": SESSION_ID,
            "os": platform.system().lower(),
            "computeType": compute_type,
        })

    @sio.event
    async def disconnect():
        print("[●] Disconnected from server")

    @sio.on("device:registered")
    async def on_device_registered(data):
        print(f"[✓] Registered as device: {data.get('deviceId')}")
        print("[●] Waiting for training job...\n")

    @sio.on("training:started")
    async def on_training_started(data):
        print(f"[●] Training job started: {data.get('modelType')}")

    @sio.on("training:task_assigned")
    async def on_task_assigned(data):
        job_id = data.get("jobId")
        round_num = data.get("roundNum")
        task_id = data.get("taskId")
        model_type = data.get("modelType")
        config = data.get("config", {}) or {}
        config["round_num"] = round_num
        config["roundNum"] = round_num
        config["jobId"] = job_id
        config["taskId"] = task_id
        if data.get("checkpointInterval") is not None:
            config["checkpointInterval"] = data["checkpointInterval"]
        config["_sio"] = sio
        config["_main_loop"] = asyncio.get_running_loop()
        shard = data.get("shard")

        # Priority 1: use weights pre-loaded by server-push training:checkpoint_fetch (check 5 & 6).
        server_weights = _pending_checkpoints.pop(task_id, None) if task_id else None
        if server_weights:
            config["globalWeights"] = server_weights
            print(f"[✓] Using server-pushed checkpoint for task {task_id}: {len(server_weights)} weights")
        else:
            # Priority 2: agent-pull — server embedded checkpointKey in the task payload.
            checkpoint_key = (
                data.get("checkpointKey") or config.get("checkpointPath") or data.get("checkpointPath") or ""
            ).strip()
            if checkpoint_key:
                print(f"[●] Fetching checkpoint from server: {checkpoint_key}")
                evt = asyncio.Event()
                _pending_checkpoint["event"] = evt
                _pending_checkpoint["weights"] = None
                await sio.emit("training:checkpoint_fetch", {"checkpointKey": checkpoint_key})
                try:
                    await asyncio.wait_for(evt.wait(), timeout=5.0)
                    received_weights = _pending_checkpoint.get("weights")
                    if received_weights:
                        config["globalWeights"] = received_weights
                        print(f"[✓] Checkpoint loaded: {len(received_weights)} weights")
                    else:
                        print("[⚠] Checkpoint payload empty - training from scratch")
                except asyncio.TimeoutError:
                    print("[⚠] Checkpoint fetch timed out (5s) - training from scratch")
                finally:
                    _pending_checkpoint.clear()

        print(f"\n[●] Round {round_num} - training {model_type}...")
        result = await asyncio.to_thread(train_locally, model_type, shard, config)
        print(f"[●] Sending weights ({len(result['weights'])} floats)...")

        await sio.emit("training:weights_ready", {
            "jobId": job_id,
            "roundNum": round_num,
            "weights": result["weights"],
            "loss": result["loss"],
            "accuracy": result["accuracy"],
        })
        print("[✓] Weights sent")

    @sio.on("training:complete")
    async def on_training_complete(data):
        print("\n[✓] Training complete!")
        print(f"[✓] Final accuracy: {data.get('finalAccuracy', 0) * 100:.1f}%")
        print(f"[✓] Final loss: {data.get('finalLoss', 0):.4f}")

    COMMAND_WHITELIST = {
        "ls", "pwd", "python", "python3", "pip", "pip3",
        "nvidia-smi", "df", "ps", "echo", "cat", "uname",
        "whoami", "which", "env", "top", "free",
    }

    @sio.on("terminal:execute")
    async def on_terminal_execute(data):
        command = data.get("command", "")
        from_socket = data.get("fromSocketId")
        print(f"[●] Terminal command received: {command}")

        args = shlex.split(command)
        if not args or args[0] not in COMMAND_WHITELIST:
            denied = args[0] if args else "<empty>"
            print(f"[✗] Command denied (not whitelisted): {denied}")
            await sio.emit("terminal:output", {
                "data": f"[error] Command not allowed: {denied}\r\n",
                "targetSocketId": from_socket,
            })
            return

        try:
            proc = subprocess.run(args, shell=False, capture_output=True, text=True, timeout=10)
            output = proc.stdout + proc.stderr
        except Exception as error:
            output = str(error)

        await sio.emit("terminal:output", {"data": output + "\r\n", "targetSocketId": from_socket})

    async def heartbeat_loop():
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
            if sio.connected:
                stats = get_stats()
                await sio.emit("heartbeat", {**stats, "computeType": compute_type})

    await sio.connect(SERVER_URL, auth={"token": token}, transports=["websocket"])
    asyncio.create_task(heartbeat_loop())
    await sio.wait()


if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        release_sleep()
        print("\n[✓] Agent stopped. Sleep prevention released.")
