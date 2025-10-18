# ====================================================
# 🧠 ReproduceAI — Makefile (Unified Stage Naming)
# ====================================================

PYTHON = .venv/bin/python
SCRIPTS_DIR = scripts
INDEX_FILE = PAPER_INDEX.json

help:
	@echo "🧠 ReproduceAI — Available commands:"
	@echo "  make init                    -> Initialize project structure"
	@echo "  make add name=\"<PaperName>\" year=<YYYY> org=\"<Org>\" stage=<stage>"
	@echo "  make list                    -> List all registered papers"
	@echo "  make status                  -> Show current reproduction progress"
	@echo "  make clean                   -> Remove temporary files"
	@echo ""
	@echo "  Valid stages:"
	@echo "     foundation      (Modern Foundation Models)"
	@echo "     representation  (Transformers & Embeddings)"
	@echo "     deep            (Deep Learning Renaissance)"
	@echo "     statistical     (Classical ML)"
	@echo "     neural          (Early Neural Origins)"

init:
	@echo "📁 Initializing ReproduceAI directory structure..."
	mkdir -p stage1_foundation stage2_representation stage3_deep_renaissance stage4_statistical stage5_neural_origins scripts paper_template
	@echo "✅ Base directories created."

add:
	@if [ -z "$(name)" ] || [ -z "$(year)" ] || [ -z "$(org)" ] || [ -z "$(stage)" ]; then \
		echo "❌ Missing arguments. Usage: make add name=\"<PaperName>\" year=<YYYY> org=\"<Org>\" stage=<stage>"; \
		exit 1; \
	fi
	@echo "➕ Adding paper: $(name) ($(year)) — $(org) [$(stage)]"
	@$(PYTHON) $(SCRIPTS_DIR)/add_paper.py --year $(year) --name "$(name)" --org "$(org)" --stage $(stage)
	@echo "✅ Done."

list:
	@echo "📄 Listing all papers in $(INDEX_FILE):"
	@$(PYTHON) -c "import json; d=json.load(open('$(INDEX_FILE)')); \
	print('\n'.join([f\"{p['year']} | {p['paper']} | {p['organization']} | {p['status']}\" for p in d['papers']]))"

status:
	@echo "📊 Current Progress by Stage:"
	@$(PYTHON) -c "import json,collections; \
d=json.load(open('PAPER_INDEX.json')); \
cnt=collections.Counter(p['stage'] for p in d['papers']); \
name_map={'Foundation':'🪐 Modern Foundation','Representation':'🔍 Representation','Deep':'🧩 Deep Renaissance','Statistical':'📊 Statistical','Neural Origins':'🧬 Neural Origins'}; \
order=['Foundation','Representation','Deep','Statistical','Neural Origins']; \
[print(f'{name_map.get(s,s):25}: {cnt.get(s,0):3d} papers') for s in order]"

clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup done."

report:
	@echo "📊 Generating summary report..."
	@$(PYTHON) -c "import json,datetime,collections; \
d=json.load(open('PAPER_INDEX.json')); \
total=len(d['papers']); \
stages=collections.Counter(p['stage'] for p in d['papers']); \
print(f'📅 Report generated: {datetime.date.today()}'); \
print(f'🧩 Total Papers: {total}'); \
[print(f'  {s:15} -> {c} papers') for s,c in stages.items()]"
