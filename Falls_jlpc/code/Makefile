# Makefile for the radar fall-detection pipeline (flat layout).
# All .py files live in this same directory. Run targets from inside Falling/code/.

PYTHON     ?= python
DATA_ROOT  ?= ..
CACHE_DIR  ?= cache
RUNS_DIR   ?= runs
SEED       ?= 17

.PHONY: help install test manifest preprocess train_vae train_binary train_multi \
        eval_vae train_vae_holdout eval_holdout clean

help:
	@echo "Targets:"
	@echo "  make install         — pip install -r requirements.txt"
	@echo "  make test            — run pytest on test_critical_invariants.py"
	@echo "  make manifest        — walk all 7 session folders, build manifest.csv"
	@echo "  make preprocess      — precompute spectrograms to $(CACHE_DIR)/"
	@echo "  make summary         — text + plots of corpus and SVM-baseline feature"
	@echo "  make train_vae       — train the conv-VAE anomaly detector"
	@echo "  make train_binary    — train the binary CNN baseline"
	@echo "  make train_multi     — train the 6-class CNN baseline"
	@echo "  make eval_vae        — evaluate the trained VAE"
	@echo "  make train_vae_holdout — train with West Cumbria held out"
	@echo "  make eval_holdout    — evaluate the holdout-trained VAE"
	@echo "  make clean           — remove cache and runs"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest test_critical_invariants.py -v

manifest:
	$(PYTHON) build_manifest.py --data_root $(DATA_ROOT) --out manifest.csv

preprocess:
	$(PYTHON) preprocess_all.py --data_root $(DATA_ROOT) --cache_dir $(CACHE_DIR)

summary:
	$(PYTHON) data_summary.py --data_root $(DATA_ROOT) --cache_dir $(CACHE_DIR) --out_dir summary

train_vae:
	$(PYTHON) train_vae.py \
		--data_root $(DATA_ROOT) --cache_dir $(CACHE_DIR) \
		--out_dir $(RUNS_DIR)/vae_default \
		--epochs 200 --batch_size 64 --lr 1e-3 \
		--latent_dim 64 --beta_kl 1.0 --warmup_epochs 20 \
		--seed $(SEED)

train_binary:
	$(PYTHON) train_baseline.py \
		--data_root $(DATA_ROOT) --cache_dir $(CACHE_DIR) \
		--out_dir $(RUNS_DIR)/baseline_binary \
		--task binary --epochs 100 --seed $(SEED)

train_multi:
	$(PYTHON) train_baseline.py \
		--data_root $(DATA_ROOT) --cache_dir $(CACHE_DIR) \
		--out_dir $(RUNS_DIR)/baseline_multi \
		--task multiclass --epochs 100 --seed $(SEED)

eval_vae:
	$(PYTHON) evaluate.py \
		--data_root $(DATA_ROOT) --cache_dir $(CACHE_DIR) \
		--ckpt $(RUNS_DIR)/vae_default/best.pt \
		--target_recall 0.95

train_vae_holdout:
	$(PYTHON) train_vae.py \
		--data_root $(DATA_ROOT) --cache_dir $(CACHE_DIR) \
		--out_dir $(RUNS_DIR)/vae_holdout_cumbria \
		--split holdout --holdout_session mar2019_cumbria \
		--epochs 200 --seed $(SEED)

eval_holdout:
	$(PYTHON) evaluate.py \
		--data_root $(DATA_ROOT) --cache_dir $(CACHE_DIR) \
		--ckpt $(RUNS_DIR)/vae_holdout_cumbria/best.pt \
		--split holdout --holdout_session mar2019_cumbria \
		--target_recall 0.95

clean:
	rm -rf $(CACHE_DIR) $(RUNS_DIR) manifest.csv __pycache__ .pytest_cache
