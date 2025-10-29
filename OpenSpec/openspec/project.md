# OpenSpec Project Overview


# 專案目標
本專案旨在建立一個垃圾郵件（spam email）分類系統，並以機器學習方法進行實作與優化。

## 近期目標
1. 建立基礎的 spam email 分類器：
	- 方法：SVM（Support Vector Machine）
	- 資料集：[sms_spam_no_header.csv](https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv)
	- 步驟：資料前處理、特徵工程、模型訓練與評估

## 後續規劃（預留空間）
2. 進一步優化與擴展：
	- 可能嘗試不同的分類方法（如 Logistic Regression、Random Forest 等）
	- 增加特徵工程、模型調參、部署等內容
	- 其他尚在規劃中

A minimal CLI tool that helps developers set up OpenSpec file structures and keep AI instructions updated. The AI tools themselves handle all the change management complexity by working directly with markdown files.

## Technology Stack
- Language: TypeScript
- Runtime: Node.js (≥20.19.0, ESM modules)
- Package Manager: pnpm
- CLI Framework: Commander.js
- User Interaction: @inquirer/prompts
- Distribution: npm package

## Project Structure
```
src/
├── cli/        # CLI command implementations
├── core/       # Core OpenSpec logic (templates, structure)
└── utils/      # Shared utilities (file operations, rollback)

dist/           # Compiled output (gitignored)
```

## Conventions
- TypeScript strict mode enabled
- Async/await for all asynchronous operations
- Minimal dependencies principle
- Clear separation of CLI, core logic, and utilities
- AI-friendly code with descriptive names

## Error Handling
- Let errors bubble up to CLI level for consistent user messaging
- Use native Error types with descriptive messages
- Exit with appropriate codes: 0 (success), 1 (general error), 2 (misuse)
- No try-catch in utility functions, handle at command level

## Logging
- Use console methods directly (no logging library)
- console.log() for normal output
- console.error() for errors (outputs to stderr)
- No verbose/debug modes initially (keep it simple)

## Testing Strategy
- Manual testing via `pnpm link` during development
- Smoke tests for critical paths only (init, help commands)
- No unit tests initially - add when complexity grows
- Test commands: `pnpm test:smoke` (when added)

## Development Workflow
- Use pnpm for all package management
- Run `pnpm run build` to compile TypeScript
- Run `pnpm run dev` for development mode
- Test locally with `pnpm link`
- Follow OpenSpec's own change-driven development process