def train_delay_recovery_from_quantized_weights(
        model,
        ckpt,
        train_loader,
        val_loader,
        test_loader,
        epochs,
        bits,
        max_delay,
        device,
        seed):

    model.load_state_dict(
        torch.load(ckpt, map_location=device)
    )

    print(f"\n=== Delay Recovery Experiment ({bits}-bit) ===")

    # Quantize weights once
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                param.copy_(quantize_tensor(param.data, bits))

    # Freeze weights
    for name, param in model.named_parameters():

        if "weight" in name:
            param.requires_grad = False

        elif "delay" in name:
            param.requires_grad = True
            with torch.no_grad():
                nn.init.uniform_(param.data, a=0.0, b=min(1.0, max_delay))

    delay_params = [
        p for n,p in model.named_parameters()
        if "delay" in n
    ]

    optimizer = torch.optim.Adam(
        delay_params,
        lr=0.1
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5)

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    best_val_acc = -1.0

    spike_monitor = monitor.OutputMonitor(model, neuron.LIFNode)

    history = {
        'train_loss': [], 'train_acc': [], 'train_spikes': [],
        'val_loss': [], 'val_acc': [], 'val_spikes': []
    }

    for epoch in range(epochs):

        model.train()

        correct = 0
        total = 0
        total_train_loss = 0
        total_train_spikes = 0

        for frames, labels in train_loader:
            functional.reset_net(model)
            optimizer.zero_grad()

            if frames.dim() == 5:
                B,T,C,H,W = frames.shape
                frames = frames.view(B,T,C*H*W)
                frames = frames.permute(1,0,2)

            elif frames.dim() == 3:
                frames = frames.permute(1,0,2)

            frames = frames.to(device)
            labels = labels.to(device)

            spike_monitor.enable()
            outputs = model(frames)

            batch_spikes = sum(r.sum().item() for r in spike_monitor.records)
            total_train_spikes += (batch_spikes / labels.size(0))
            spike_monitor.records.clear()
            spike_monitor.disable()

            mean_out = outputs.mean(dim=0)

            loss = criterion(mean_out, labels)

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                for n,p in model.named_parameters():
                    if "delay" in n:
                        p.clamp_(0,max_delay)

            pred = mean_out.argmax(dim=1)

            correct += (pred == labels).sum().item()
            total += labels.numel()


            total_train_loss += loss.item()

        epoch_train_acc = 100 * correct / total
        epoch_train_loss = total_train_loss / len(train_loader)
        epoch_train_avg_spikes = total_train_spikes / len(train_loader)

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['train_spikes'].append(epoch_train_avg_spikes)

        # Validation

        model.eval()

        total_val_loss, val_correct, val_total, total_val_spikes = 0, 0, 0, 0

        with torch.no_grad():

            for frames, labels in val_loader:
                functional.reset_net(model)
                if frames.dim() == 5:
                    B,T,C,H,W = frames.shape
                    frames = frames.view(B,T,C*H*W)
                    frames = frames.permute(1,0,2)

                elif frames.dim() == 3:
                    frames = frames.permute(1,0,2)

                frames = frames.to(device)
                labels = labels.to(device)

                spike_monitor.enable()

                outputs = model(frames)

                batch_val_spikes = sum(r.sum().item() for r in spike_monitor.records)
                total_val_spikes += (batch_val_spikes / labels.size(0))
                spike_monitor.records.clear()
                spike_monitor.disable()

                outputs = outputs.mean(dim=0)
                v_loss = criterion(outputs, labels)
                total_val_loss += v_loss.item()
                pred = outputs.argmax(dim=1)

                val_correct += (pred == labels).sum().item()
                val_total += labels.numel()

        epoch_val_loss = total_val_loss / len(val_loader)
        epoch_val_acc = (val_correct / val_total) * 100
        epoch_val_avg_spikes = total_val_spikes / len(val_loader)

        scheduler.step(epoch_val_acc)

        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_spikes'].append(epoch_val_avg_spikes)

        if epoch_val_acc > best_val_acc:
          best_val_acc = epoch_val_acc
          os.makedirs(f"./outputs_quant_delay_seed{seed}", exist_ok=True)
          torch.save(
              model.state_dict(),
              f"./outputs_quant_delay_seed{seed}/best_delay_recovery_{bits}bits.pth"
          )


        print(
            f"Epoch {epoch+1}/{epochs} "
            f"| Train Acc = {epoch_train_acc:.2f}% "
            f"| Train Loss = {epoch_train_loss:.4f}"
            f"| Val Acc = {epoch_val_acc:.2f}% "
            f"| Val Loss = {epoch_val_loss:.4f}"
            f"| Best Val = {best_val_acc:.2f}%"
        )

    log_path = f"./outputs_quant_delay_seed{seed}/quant_delay_logs_{bits}bit.txt"
    os.makedirs(f"./outputs_quant_delay_seed{seed}", exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Epoch,Train_Loss,Train_Acc,Train_Spikes,Val_Loss,Val_Acc,Val_Spikes\n")
        for i in range(len(history['train_loss'])):
            f.write(
                f"{i+1},{history['train_loss'][i]:.4f},{history['train_acc'][i]:.2f},{history['train_spikes'][i]:.1f},"
                f"{history['val_loss'][i]:.4f},{history['val_acc'][i]:.2f},{history['val_spikes'][i]:.1f}\n"
            )
    print(f"All learning logs archived at '{log_path}'.")

    # Final Test

    model.load_state_dict(
      torch.load(
          f"./outputs_quant_delay_seed{seed}/best_delay_recovery_{bits}bits.pth",
          map_location=device
      )
    )

    model.eval()

    total_test_loss = 0
    test_correct = 0
    test_total = 0
    total_test_spikes = 0

    with torch.no_grad():

        for frames, labels in test_loader:
            functional.reset_net(model)
            if frames.dim() == 5:
                B,T,C,H,W = frames.shape
                frames = frames.view(B,T,C*H*W)
                frames = frames.permute(1,0,2)

            elif frames.dim() == 3:
                frames = frames.permute(1,0,2)

            frames = frames.to(device)
            labels = labels.to(device)

            spike_monitor.enable()

            outputs = model(frames)

            batch_test_spikes = sum(r.sum().item() for r in spike_monitor.records)
            total_test_spikes += (batch_test_spikes / labels.size(0))
            spike_monitor.records.clear()
            spike_monitor.disable()

            mean_out = outputs.mean(dim=0)
            test_loss_batch = criterion(mean_out, labels)
            pred = mean_out.argmax(dim=1)

            total_test_loss += test_loss_batch.item()
            test_correct += (pred == labels).sum().item()
            test_total += labels.numel()

    test_loss = total_test_loss / len(test_loader)
    test_acc = 100 * test_correct / test_total
    final_test_spikes = total_test_spikes / len(test_loader)
    spike_monitor.remove_hooks()
    print(
        f"\nRecovered Accuracy ({bits}-bit + Delay Learning): "
        f"{test_acc:.2f}% | "
        f"Loss: {test_loss:.4f} | "
        f"Avg Spike Count: {final_test_spikes:.1f}"
    )
    with open(
        f"./outputs_quant_delay_seed{seed}/result_{bits}bit.txt",
        "w"
    ) as f:
        f.write(f"Best Val Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Final Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Final Test Loss: {test_loss:.4f}\n")
        f.write(f"Final Test Avg Spikes: {final_test_spikes:.4f}\n")
    return test_loss, test_acc, final_test_spikes
