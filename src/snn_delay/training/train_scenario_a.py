import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, monitor, neuron
import os

def train_weight_only_fixed_delay_snn(model, train_loader, val_loader, test_loader, epochs, device, seed):

    os.makedirs(f"./outputs_fixed_delay_seed{seed}", exist_ok=True)

    print("\n1- Starting Weight-Only Learning...")

    for param in model.parameters():
        param.requires_grad = True

    weight_parameters = [p for n, p in model.named_parameters() if 'weight' in n]
    delay_parameters = [p for n, p in model.named_parameters() if 'delay' in n]

    for p in delay_parameters:
        p.requires_grad = False


    optimizer = torch.optim.Adam([{'params': weight_parameters, 'lr': 0.01, 'weight_decay': 4e-4}])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.2,
        patience=7,
        cooldown=1,
        min_lr=1e-5
    )

    criterion = nn.CrossEntropyLoss()

    spike_monitor = monitor.OutputMonitor(model, neuron.LIFNode)
    model.to(device)

    history = {
        'train_loss': [], 'train_acc': [], 'train_spikes': [],
        'val_loss': [], 'val_acc': [], 'val_spikes': []
    }

    best_val_acc = 0.0
    best_val_loss = 100.0
    print("\nStarting Training Loop (Weights Tuning)...")

    try:
        for epoch in range(epochs):
            # Train Phase
            model.train()
            total_train_loss, train_correct, train_total, total_train_spikes = 0, 0, 0, 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                functional.reset_net(model)
                optimizer.zero_grad()

                if inputs.dim() == 5:
                    B, T, C, H, W = inputs.shape
                    inputs = inputs.view(B, T, C * H * W).permute(1, 0, 2)
                elif inputs.dim() == 3:
                    inputs = inputs.permute(1, 0, 2)

                inputs, targets = inputs.to(device), targets.to(device)

                spike_monitor.enable()
                out_spikes = model(inputs)

                batch_spikes = sum(r.sum().item() for r in spike_monitor.records)
                total_train_spikes += (batch_spikes / targets.size(0))
                spike_monitor.records.clear()
                spike_monitor.disable()

                mean_firing_rate = out_spikes.mean(dim=0)
                loss = criterion(mean_firing_rate, targets)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(mean_firing_rate, dim=1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()


                total_train_loss += loss.item()

            epoch_train_loss = total_train_loss / len(train_loader)
            epoch_train_acc = (train_correct / train_total) * 100
            epoch_train_avg_spikes = total_train_spikes / len(train_loader)

            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            history['train_spikes'].append(epoch_train_avg_spikes)

            # Validation Phase
            model.eval()
            total_val_loss, val_correct, val_total, total_val_spikes = 0, 0, 0, 0

            with torch.no_grad():
                for frames, labels in val_loader:
                    functional.reset_net(model)
                    if frames.dim() == 5:
                        B, T, C, H, W = frames.shape
                        frames = frames.view(B, T, C * H * W).permute(1, 0, 2)
                    elif frames.dim() == 3:
                        frames = frames.permute(1, 0, 2)

                    frames, labels = frames.to(device), labels.to(device)

                    spike_monitor.enable()
                    outputs = model(frames)

                    batch_val_spikes = sum(r.sum().item() for r in spike_monitor.records)
                    total_val_spikes += (batch_val_spikes / labels.size(0))
                    spike_monitor.records.clear()
                    spike_monitor.disable()

                    mean_out = outputs.mean(dim=0)
                    v_loss = criterion(mean_out, labels)
                    total_val_loss += v_loss.item()

                    pred = mean_out.argmax(dim=1)
                    val_correct += (pred == labels).sum().item()
                    val_total += labels.numel()



            epoch_val_loss = total_val_loss / len(val_loader)
            epoch_val_acc = (val_correct / val_total) * 100
            epoch_val_avg_spikes = total_val_spikes / len(val_loader)

            scheduler.step(epoch_val_acc)

            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc)
            history['val_spikes'].append(epoch_val_avg_spikes)

            print(
                f"Epoch {epoch+1:02d}/{epochs} | "
                f"Train Loss={epoch_train_loss:.4f}, Acc={epoch_train_acc:.2f}%, Spikes={epoch_train_avg_spikes:.1f} || "
                f"Val Loss={epoch_val_loss:.4f}, Acc={epoch_val_acc:.2f}%, Spikes={epoch_val_avg_spikes:.1f}"
            )

            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(model.state_dict(), f"./outputs_fixed_delay_seed{seed}/best_val_acc_model.pth")
                print(f"--> [SAVE] Checkpoint Archived! Best Val Acc: {best_val_acc:.2f}%")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    log_path = f"./outputs_fixed_delay_seed{seed}/fixed_delay_logs.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Epoch,Train_Loss,Train_Acc,Train_Spikes,Val_Loss,Val_Acc,Val_Spikes\n")
        for i in range(len(history['train_loss'])):
            f.write(
                f"{i+1},{history['train_loss'][i]:.4f},{history['train_acc'][i]:.2f},{history['train_spikes'][i]:.1f},"
                f"{history['val_loss'][i]:.4f},{history['val_acc'][i]:.2f},{history['val_spikes'][i]:.1f}\n"
            )
    print(f"All learning logs archived at '{log_path}'.")

    # Test Phase
    print("\n=== Training Finished. Loading Best Acc Model for Final Test Evaluation... ===")
    if os.path.exists(f"./outputs_fixed_delay_seed{seed}/best_val_acc_model.pth"):
        model.load_state_dict(torch.load(f"./outputs_fixed_delay_seed{seed}/best_val_acc_model.pth"))

    model.eval()
    total_test_loss, test_correct, test_total, total_test_spikes = 0, 0, 0, 0

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
            spike_monitor.records.clear()
            spike_monitor.disable()

            mean_out = outputs.mean(dim=0)
            t_loss = criterion(mean_out, labels)
            pred = mean_out.argmax(dim=1)
            total_test_loss += t_loss.item()
            test_correct += (pred == labels).sum().item()
            test_total += labels.numel()


    final_test_loss = total_test_loss / len(test_loader)
    final_test_acc = (test_correct / test_total) * 100
    final_test_spikes = total_test_spikes / len(test_loader)

    print(f"\n🚀 [FINAL RESULTS - FIXED DELAY] 🚀")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Final Test Loss (Unseen Data): {final_test_loss:.4f}")
    print(f"Final Test Accuracy (Unseen Data): {final_test_acc:.2f}%")
    print(f"Final Test Average Spike Count: {final_test_spikes:.1f}")

    with open(f"./outputs_fixed_delay_seed{seed}/final_test_acc_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Best Val Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Final Test Loss: {final_test_loss:.4f}\n")
        f.write(f"Final Test Accuracy: {final_test_acc:.2f}%\n")
        f.write(f"Final Test Average Spikes: {final_test_spikes:.1f}\n")
