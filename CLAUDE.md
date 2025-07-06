# Plumial AI Agent Instructions

This file provides a framework for AI agents working with the Plumial library. It allows users to customize how AI assistants interact with their specific fork or deployment of the codebase.

## User Customization

Users of the Plumial library can add their own AI agent rules and instructions by creating a rules directory structure:

```
rules/plumial/CLAUDE.md
```

This approach allows for:
- **Personal workflow preferences**: Custom coding styles, testing approaches, documentation standards
- **Project-specific requirements**: Domain expertise, mathematical conventions, validation rules
- **Team collaboration**: Shared AI interaction patterns across contributors
- **Experimental guidelines**: Rules for exploring new mathematical territory safely

## Important Guidelines

**In most cases, this root CLAUDE.md file should NOT be updated.** Instead:

1. Create your personalized rules file at `rules/plumial/CLAUDE.md`
2. Add project-specific instructions, mathematical conventions, and workflow preferences there
3. Keep this root file as a stable reference for the general library structure

The `rules/` directory is excluded from version control (see `.gitignore`), ensuring that personal AI instructions remain private while the core library stays clean and collaborative.

## For AI Agents

When working with this codebase:
1. First check for user-specific rules at `rules/plumial/CLAUDE.md`
2. Respect any mathematical conventions and terminology preferences found there
3. Follow the established patterns for polynomial representations, cycle analysis, and symbolic computation
4. Maintain the rigorous mathematical foundations that underpin all Plumial operations

This approach ensures that AI assistance remains both mathematically sound and personally tailored to each user's research style and preferences.