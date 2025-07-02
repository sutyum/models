# Coding Style Guide for Claude

## 🎯 Core Principles

### Minimalism First
- **Less is more**: Every line of code should justify its existence
- **Use existing tools**: Leverage DSPy primitives and built-in capabilities instead of reinventing
- **Avoid sprawl**: Consolidate related functionality into single, well-designed modules
- **No fluff**: Remove comments, docstrings, and code that doesn't add value

### Clean Code Practices
- **Single responsibility**: Each function/class should do one thing well
- **DRY (Don't Repeat Yourself)**: Eliminate code duplication ruthlessly
- **YAGNI (You Aren't Gonna Need It)**: Don't add features "just in case"
- **Flat is better than nested**: Avoid deep nesting and complex hierarchies

## 📦 Package Management

### ALWAYS Use UV
```bash
# ✅ CORRECT
uv run python main.py
uv add dspy-ai
uv sync

# ❌ WRONG - Never use these
python main.py
pip install dspy-ai
```

### Environment Setup
```bash
# Initialize project
uv venv
uv add dspy-ai click

# Run commands
uv run python main.py optimize
uv run ./run.sh evaluate
```

## 🏗️ DSPy Best Practices

### Model Configuration
```python
# Default model and settings
MODEL = "together_ai/Qwen/Qwen3-235B-A22B-fp8-tput"
lm = dspy.LM(MODEL, api_key=api_key, max_tokens=30000)
```

### Use DSPy Primitives
```python
# ✅ GOOD - Let DSPy handle complexity
class QA(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()

model = dspy.ChainOfThought(QA)

# ❌ BAD - Custom prompt engineering
def answer_question(question):
    prompt = f"Question: {question}\nThink step by step..."
    return call_llm(prompt)
```

### Leverage Built-in Features
```python
# ✅ GOOD - Use DSPy's Evaluate
evaluator = Evaluate(devset=data, metric=metric, num_threads=8)
score = evaluator(model)

# ❌ BAD - Custom evaluation loops
for example in data:
    pred = model(example.question)
    scores.append(metric(example, pred))
```

## 📁 File Organization

### Preferred Structure
```
project/
├── main.py          # Single entry point with CLI
├── models.py        # DSPy models/signatures (if needed)
├── run.sh           # Simple bash commands
├── @CLAUDE.md       # This file
└── data/           # Data files
```

### Avoid
- Multiple scripts doing similar things
- Deep directory hierarchies
- Separate files for train/eval/optimize (put in main.py)
- Complex deployment infrastructure

## ✨ Code Examples

### Minimal CLI Application
```python
#!/usr/bin/env python3
"""One-line description."""

import click
import dspy

@click.command()
@click.option('--input', help='Input file')
def main(input):
    """Main entry point."""
    # Core logic here
    pass

if __name__ == "__main__":
    main()
```

### DSPy Module Pattern
```python
class MinimalModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought("input -> output")
    
    def forward(self, input):
        return self.predictor(input=input)
```

## 🚫 What to Avoid

1. **No verbose docstrings** - Code should be self-documenting
2. **No complex class hierarchies** - Prefer composition
3. **No custom training loops** - Use DSPy optimizers
4. **No manual prompt templates** - Use DSPy signatures
5. **No pip or python commands** - Always use uv
6. **No emojis in output** - Keep output clean and professional
7. **Minimal printing** - Only essential information

## 🔧 Refactoring Checklist

When reviewing/refactoring code:
- [ ] Can this be done with DSPy primitives?
- [ ] Is there duplicate code to consolidate?
- [ ] Can multiple files be merged?
- [ ] Are we using `uv run` everywhere?
- [ ] Is every line of code necessary?
- [ ] Can this be simpler?

## 📝 Example Refactor

### Before (❌ Complex)
```python
# train.py
def train_model():
    # 100 lines of custom training

# evaluate.py  
def evaluate_model():
    # 100 lines of custom evaluation

# optimize.py
def optimize_prompts():
    # 100 lines of custom optimization
```

### After (✅ Minimal)
```python
# main.py
@click.group()
def cli():
    pass

@cli.command()
def optimize():
    optimizer = BootstrapFewShot(metric=metric)
    optimized = optimizer.compile(model, trainset=data)
    optimized.save("model.json")

@cli.command()
def evaluate():
    evaluator = Evaluate(devset=data, metric=metric)
    print(f"Score: {evaluator(model):.1%}")
```

## 🎓 Remember

> "Perfection is achieved not when there is nothing more to add, but when there is nothing left to take away." - Antoine de Saint-Exupéry

Always strive for elegant simplicity. If you can't explain what a piece of code does in one sentence, it's too complex.