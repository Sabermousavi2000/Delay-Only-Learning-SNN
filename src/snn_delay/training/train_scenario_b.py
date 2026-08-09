import os
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, monitor

def quantize_tensor(tensor, bits=8):
    if bits == 32:
        return tensor.clone()
    qmin = -(2 ** (bits - 1)) + 1
    qmax = (2 ** (bits - 1)) - 1
    max_val = torch.max(torch.abs(tensor))
    if max_val == 0:
        return tensor.clone()
    scale = max_val / qmax
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, qmin, qmax)
    return quantized * scale

def run_test_for_checkpoint(model_fn, test_loader, checkpoint_path, bits, device):
    model = model_fn()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    if bits < 32:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name:
                    param.copy_(quantize_tensor(param.data, bits=bits))

    spike_monitor = monitor.OutputMonitor(model, neuron.LIFNode)
    test_correct, test_total, total_test_loss, total_test_spikes = 0, 0, 0, 0

    with torch.no_grad():
        for frames, labels in test_loader:
            functional.reset_net(model)
            if frames.dim() == 5:
                B, T, C, H, W = frames.shape
                frames = frames.view(B, T, C * H * W).permute(1, 0, 2)
            elif frames.dim() == 3:
                frames = frames.permute(1, 0, 2)

            frames, labels = frames.to(device), labels.to(device)

            spike_monitor.enable()
            outputs = model(frames)

            batch_test_spikes = sum(r.sum().item() for r in spike_monitor.records)
            total_test_spikes += (batch_test_spikes / labels.size(0))

            spike_monitor.disable()
            spike_monitor.records.clear()

            mean_out = outputs.mean(dim=0)
            t_loss = criterion(mean_out, labels)
            total_test_loss += t_loss.item()
            pred = mean_out.argmax(dim=1)
            test_correct += (pred == labels).sum().item()
            test_total += labels.numel()


    final_test_loss = total_test_loss / len(test_loader)
    final_test_acc = (test_correct / test_total) * 100
    final_test_spikes = total_test_spikes / len(test_loader)
    return final_test_loss, final_test_acc, final_test_spikes

def compare_joint_vs_fixed_quantization(model_fn, ckpt, test_loader, device='cuda', seed=42):
    fixed_ckpt = ckpt

    bits_options = [32, 8, 4, 2]
    results = {"Fixed-Delay": {}}

    print("=== Starting Comprehensive Quantization Comparison ===")

    for bits in bits_options:
        print(f"\n[Evaluating {bits}-bit Precision...]")


        if os.path.exists(fixed_ckpt):
            loss, acc, spks = run_test_for_checkpoint(model_fn, test_loader, fixed_ckpt, bits, device)
            results["Fixed-Delay"][bits] = (loss, acc, spks)
            print(f"-> Fixed-Delay Model Acc: {acc:.2f}%"
            f" Loss: {loss:.2f}"
            f" avg spikes: {spks:.2f}")
        else:
            results["Fixed-Delay"][bits] = None

    print("\n" + "="*80)
    print("FINAL QUANTIZATION ABLATION TABLE")
    print("="*80)
    print(f"{'Bits':<10}{'Loss':<12}{'Acc (%)':<12}{'Avg Spikes':<12}")
    print("-"*80)

    for bits in bits_options:
        result = results["Fixed-Delay"][bits]

        if result is not None:
            loss, acc, spks = result
            print(f"{bits:<10}{loss:<12.4f}{acc:<12.2f}{spks:<12.2f}")
        else:
            print(f"{bits:<10}{'N/A':<12}{'N/A':<12}{'N/A':<12}")

    print("="*80)

    with open(f"./outputs_fixed_delay_seed{seed}/quantization_comparison_report.txt",
              "w", encoding="utf-8") as f:
        f.write("Bits,Loss,Accuracy,AvgSpikes\n")

        for bits in bits_options:
            result = results["Fixed-Delay"][bits]

            if result is not None:
                loss, acc, spks = result
                f.write(f"{bits},{loss:.6f},{acc:.2f},{spks:.2f}\n")
            else:
                f.write(f"{bits},N/A,N/A,N/A\n")
