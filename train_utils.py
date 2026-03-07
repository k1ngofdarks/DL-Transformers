from __future__ import annotations

import copy
import math

import matplotlib.pyplot as plt
import sacrebleu
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import clear_output
from tqdm.auto import tqdm


AMP_DTYPE = torch.float16


def greedy_decode(model, src, src_key_padding_mask, max_len, bos_id, eos_id):
    model_for_decode = model
    device = src.device
    batch_size = src.size(0)

    src_emb = model_for_decode.pos_enc(model_for_decode.src_embed(src))
    memory = model_for_decode.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    ys = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_len = ys.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(ys.device)

        tgt_emb = model_for_decode.pos_enc(model_for_decode.tgt_embed(ys))
        out = model_for_decode.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        logits = model_for_decode.fc_out(out)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)

        if (next_token.squeeze(1) == eos_id).all():
            break

    return ys


def _beam_search_decode_single(model, src, src_key_padding_mask, max_len, bos_id, eos_id, beam_size=4, length_penalty=0.7):
    model_for_decode = model
    device = src.device

    src_emb = model_for_decode.pos_enc(model_for_decode.src_embed(src))
    memory = model_for_decode.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    beams = [(torch.tensor([bos_id], device=device, dtype=torch.long), 0.0, False)]

    for _ in range(max_len - 1):
        candidates = []

        for seq, score, finished in beams:
            if finished:
                candidates.append((seq, score, True))
                continue

            ys = seq.unsqueeze(0)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(device)

            tgt_emb = model_for_decode.pos_enc(model_for_decode.tgt_embed(ys))
            out = model_for_decode.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            logits = model_for_decode.fc_out(out[:, -1, :])
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

            topk_log_probs, topk_ids = torch.topk(log_probs, k=beam_size)
            for next_log_prob, next_id in zip(topk_log_probs, topk_ids):
                next_id = int(next_id.item())
                new_seq = torch.cat([seq, torch.tensor([next_id], device=device)])
                new_score = score + float(next_log_prob.item())
                new_finished = next_id == eos_id
                candidates.append((new_seq, new_score, new_finished))

        def rank_key(item):
            seq, score, _ = item
            norm = ((5 + seq.size(0)) / 6) ** length_penalty
            return score / norm

        beams = sorted(candidates, key=rank_key, reverse=True)[:beam_size]

        if all(finished for _, _, finished in beams):
            break

    best_seq = max(
        beams,
        key=lambda item: item[1] / (((5 + item[0].size(0)) / 6) ** length_penalty),
    )[0]
    return best_seq


def beam_decode(model, src, src_key_padding_mask, max_len, bos_id, eos_id, beam_size=4, length_penalty=0.7):
    decoded = []
    for i in range(src.size(0)):
        best_seq = _beam_search_decode_single(
            model=model,
            src=src[i : i + 1],
            src_key_padding_mask=src_key_padding_mask[i : i + 1],
            max_len=max_len,
            bos_id=bos_id,
            eos_id=eos_id,
            beam_size=beam_size,
            length_penalty=length_penalty,
        )
        decoded.append(best_seq)

    return torch.nn.utils.rnn.pad_sequence(decoded, batch_first=True, padding_value=eos_id)


def batch_translate(
    model,
    texts,
    src_tokenizer,
    tgt_tokenizer,
    device,
    batch_size=32,
    max_len=80,
    decode_strategy="beam",
    beam_size=4,
    length_penalty=0.7,
):
    model.eval()
    all_predictions = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        input_ids = [src_tokenizer.encode(text) for text in batch_texts]
        src_tensor = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids[:max_len]) for ids in input_ids],
            batch_first=True,
            padding_value=src_tokenizer.word2id["<pad>"],
        ).to(device)

        src_mask = src_tensor == src_tokenizer.word2id["<pad>"]

        with torch.no_grad():
            if decode_strategy == "beam":
                out_ids = beam_decode(
                    model,
                    src_tensor,
                    src_mask,
                    max_len,
                    bos_id=tgt_tokenizer.word2id["<bos>"],
                    eos_id=tgt_tokenizer.word2id["<eos>"],
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                )
            else:
                out_ids = greedy_decode(
                    model,
                    src_tensor,
                    src_mask,
                    max_len,
                    bos_id=tgt_tokenizer.word2id["<bos>"],
                    eos_id=tgt_tokenizer.word2id["<eos>"],
                )

        for ids in out_ids:
            all_predictions.append(tgt_tokenizer.decode(ids.tolist()))

    return all_predictions


def train_model(
    model,
    train_loader,
    val_loader,
    src_tokenizer,
    tgt_tokenizer,
    val_src_texts,
    val_tgt_texts,
    epochs=10,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    amp_dtype=AMP_DTYPE,
    comet_experiment=None,
):
    pad_id_src = src_tokenizer.word2id["<pad>"]
    pad_id_tgt = tgt_tokenizer.word2id["<pad>"]

    base_lr = 3e-4
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)

    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        min_lr_ratio = 0.02
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id_tgt, label_smoothing=0.1)

    if comet_experiment is not None:
        comet_experiment.log_parameters(
            {
                "epochs": epochs,
                "base_lr": base_lr,
                "warmup_steps": warmup_steps,
                "optimizer": "AdamW",
                "weight_decay": 1e-4,
                "label_smoothing": 0.1,
                "batch_size": getattr(train_loader, "batch_size", None),
                "amp_dtype": str(amp_dtype),
            }
        )

    history_train_loss, history_val_loss, history_bleu = [], [], []
    global_step = 0
    best_bleu = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        valid_updates = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - train", leave=False)
        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(src, tgt_input, pad_id_src=pad_id_src, pad_id_tgt=pad_id_tgt)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_expected.reshape(-1))

            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss at step {global_step}: {loss.item()}. Skipping batch.")
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if not torch.isfinite(grad_norm):
                print(f"[WARN] Non-finite grad norm at step {global_step}. Skipping optimizer step.")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            valid_updates += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.7f}",
                gnorm=f"{float(grad_norm):.2f}",
            )

        train_loss = epoch_loss / max(1, valid_updates)
        history_train_loss.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - val", leave=False):
                src = src.to(device)
                tgt = tgt.to(device)

                tgt_input = tgt[:, :-1]
                tgt_expected = tgt[:, 1:]

                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    logits = model(src, tgt_input, pad_id_src=pad_id_src, pad_id_tgt=pad_id_tgt)
                    loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_expected.reshape(-1))

                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        history_val_loss.append(val_loss)

        predictions = batch_translate(model, val_src_texts[:200], src_tokenizer, tgt_tokenizer, device)
        bleu = sacrebleu.corpus_bleu(predictions, [val_tgt_texts[:200]]).score
        history_bleu.append(bleu)

        if bleu > best_bleu:
            best_bleu = bleu
            best_state = copy.deepcopy(model.state_dict())

        if comet_experiment is not None:
            comet_experiment.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_bleu": bleu,
                },
                step=epoch,
            )

        clear_output(wait=True)
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(history_train_loss, label="Train Loss", color="blue", marker="o")
        ax[0].plot(history_val_loss, label="Val Loss", color="orange", marker="o")
        ax[0].set_title("Loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
        ax[0].grid()

        ax[1].plot(history_bleu, label="Val BLEU", color="green", marker="o")
        ax[1].set_title("BLEU")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Score")
        ax[1].legend()
        ax[1].grid()

        plt.show()
        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val BLEU: {bleu:.2f} | Best BLEU: {best_bleu:.2f}"
        )

        if comet_experiment is not None and epoch % 3 == 0:
            torch.save(model.state_dict(), f"model_{epoch}.pth")
            comet_experiment.log_model("my_model", f"model_{epoch}.pth")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best checkpoint with BLEU={best_bleu:.2f}")


def translate_and_save(
    model,
    src_sentences,
    src_tokenizer,
    tgt_tokenizer,
    device,
    filename="translate.en",
    batch_size=128,
    comet_experiment=None,
):
    model.eval()
    results = []

    for i in tqdm(range(0, len(src_sentences), batch_size)):
        batch = src_sentences[i : i + batch_size]
        translated_batch = batch_translate(
            model,
            batch,
            src_tokenizer,
            tgt_tokenizer,
            device,
            batch_size=batch_size,
            max_len=100,
        )
        results.extend(translated_batch)

    with open(filename, "w", encoding="utf-8") as file:
        for line in results:
            file.write(line + "\n")

    if comet_experiment is not None:
        comet_experiment.log_asset(filename, file_name=filename)
        sample_rows = [[src_sentences[i], results[i]] for i in range(min(50, len(results)))]
        comet_experiment.log_table(
            "translation_samples.json",
            tabular_data=sample_rows,
            headers=["source_de", "pred_en"],
        )
        comet_experiment.log_metric("translations_count", len(results))

    print(f"Well done: {filename}")
