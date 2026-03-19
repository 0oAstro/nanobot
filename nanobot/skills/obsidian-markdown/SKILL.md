---
name: obsidian-markdown
description: Create and edit Obsidian Flavored Markdown with wikilinks, embeds, callouts, properties, and other Obsidian-specific syntax. Use when working with .md files in Obsidian, or when the user mentions wikilinks, callouts, frontmatter, tags, embeds, or Obsidian notes.
always: true
---

# Obsidian Flavored Markdown Skill

Create and edit valid Obsidian Flavored Markdown. Obsidian extends CommonMark and GFM with wikilinks, embeds, callouts, properties, comments, and other syntax. This skill covers only Obsidian-specific extensions. Standard Markdown is assumed knowledge.

## Workflow

1. Add frontmatter with properties at the top of the file.
2. Use standard Markdown for structure plus Obsidian-specific syntax below.
3. Use `[[wikilinks]]` for notes within the vault and normal Markdown links for external URLs.
4. Use embeds with `![[...]]` when inline inclusion is useful.
5. Use callouts for highlighted information.

Read the reference files when needed:
- `references/PROPERTIES.md`
- `references/EMBEDS.md`
- `references/CALLOUTS.md`

## Internal Links

```markdown
[[Note Name]]
[[Note Name|Display Text]]
[[Note Name#Heading]]
[[Note Name#^block-id]]
[[#Heading in same note]]
```

## Embeds

```markdown
![[Note Name]]
![[Note Name#Heading]]
![[image.png]]
![[document.pdf#page=3]]
```

## Callouts

```markdown
> [!note]
> Basic callout.

> [!warning] Custom Title
> Callout with a custom title.
```

## Properties

```yaml
---
title: My Note
tags:
  - project
  - active
aliases:
  - Alternate Name
---
```

## Tags

```markdown
#tag
#nested/tag
```

## Comments

```markdown
Visible %%hidden%% text.
```

## Highlighting

```markdown
==Highlighted text==
```
