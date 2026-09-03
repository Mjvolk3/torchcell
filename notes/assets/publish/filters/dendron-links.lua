--- notes/assets/publish/filters/dendron-links.lua
---
--- Render Dendron wikilinks as something a reader can act on ON PAPER.
---
--- Without this, pandoc emits `[[torchcell.metabolism.flux_layer]]` verbatim, so a note
--- PDF carries raw brackets where a reference belongs. A wikilink cannot be clicked in
--- print, so what a reader needs is the note's human title plus the dendron path they
--- would type to open it.
---
--- Forms handled, all three of them:
---   [[note.path]]
---   [[Display Text|note.path]]
---   [[Display Text|dendron://vault/note.path#anchor]]
---
--- Rendering:
---   * piped        -> Display Text (`note.path`)
---   * bare         -> *Title* (`note.path`), the title read from the target note's
---                     YAML frontmatter
---   * bare, where the title is only the mangled last path segment ("Gem_audit" for
---     `...scripts.gem_audit`) -> `note.path` alone, because the title adds nothing and
---     saying it twice reads as noise
---   * target absent from disk -> `note.path` alone, never a build failure
---
--- WHY AN Inlines FILTER AND NOT A STRING SUBSTITUTION. The note this was written for is
--- heavy with LaTeX, code fences and inline code, all of which contain brackets. Working
--- on the parsed document makes that safe by construction rather than by regex: Math,
--- Code, CodeBlock and RawInline are their own AST nodes, so the scan below never sees
--- inside them, and a run of text is only joined across Str/Space/SoftBreak. A `[[` in a
--- shell snippet, or an `[[Sv]_i]` inside `$...$`, is structurally out of reach.

local NOTES_DIR = "."
do
  local script = PANDOC_SCRIPT_FILE
  if script then
    local dir = script:match("^(.*)/[^/]*$")
    if dir then
      -- filters/ -> publish/ -> assets/ -> notes/
      NOTES_DIR = dir .. "/../../.."
    end
  end
end

local title_cache = {}

--- Read `title:` out of a note's YAML frontmatter. Returns nil when the note is absent.
local function note_title(path)
  local cached = title_cache[path]
  if cached ~= nil then
    if cached == false then
      return nil
    end
    return cached
  end
  local handle = io.open(NOTES_DIR .. "/" .. path .. ".md", "r")
  if not handle then
    handle = io.open(path .. ".md", "r")
  end
  if not handle then
    title_cache[path] = false
    return nil
  end
  local title = nil
  local first = handle:read("*l")
  if first and first:match("^%-%-%-%s*$") then
    for line in handle:lines() do
      if line:match("^%-%-%-%s*$") then
        break
      end
      local value = line:match("^title:%s*(.-)%s*$")
      if value then
        title = value
        break
      end
    end
  end
  handle:close()
  if title then
    -- strip a matched pair of YAML quotes
    title = title:match('^"(.*)"$') or title:match("^'(.*)'$") or title
    -- the house style bans em-dashes, and a title lifted out of frontmatter would
    -- otherwise smuggle one into a built PDF
    title = title:gsub("\226\128\148", " -- "):gsub("\226\128\147", "--")
    title = title:gsub("%s+", " "):gsub("^%s*(.-)%s*$", "%1")
    if title == "" then
      title = nil
    end
  end
  title_cache[path] = title or false
  return title
end

local function slug(s)
  return (s:lower():gsub("[^%a%d]", ""))
end

--- The path, set in typewriter, WITH somewhere to break.
---
--- A dendron path runs to about fifty characters, and typewriter text never hyphenates, so
--- plain `\texttt` gives the line breaker no option at all: the path lands whole, runs into
--- the margin, and LaTeX reports Overfull \hbox and still exits 0. A zero-width
--- `\allowbreak` after each separator is the fix that costs nothing. It must be zero-width
--- rather than a hyphenation point, because a dendron path already contains hyphens and an
--- inserted one would change what the path says.
local function tex_path(target)
  local esc = target:gsub("\\", "\\textbackslash{}")
  esc = esc:gsub("([{}$&#%%_])", "\\%1")
  esc = esc:gsub("~", "\\textasciitilde{}")
  esc = esc:gsub("%^", "\\textasciicircum{}")
  esc = esc:gsub("(\\?[%.%-/#])", "%1\\allowbreak{}")
  esc = esc:gsub("(\\_)", "%1\\allowbreak{}")
  return "\\texttt{" .. esc .. "}"
end

local function path_node(target)
  if FORMAT and FORMAT:match("latex") then
    return pandoc.RawInline("latex", tex_path(target))
  end
  return pandoc.Code(target)
end

--- Build the inlines that replace one `[[...]]`.
local function render(body)
  local display, target = body:match("^(.-)|(.*)$")
  if not target then
    target = body
    display = nil
  end
  target = target:gsub("^%s*(.-)%s*$", "%1")
  -- pandoc's smart extension runs in the READER, so a heading anchor like
  -- `#20250510---inspecting-data-in-gograph` reaches this filter with its hyphen runs
  -- already collapsed into em/en dashes. Undo that: a dendron path is ASCII by
  -- construction, and a mangled anchor is a path that does not resolve.
  target = target:gsub("\226\128\148", "---"):gsub("\226\128\147", "--")
  target = target:gsub("^dendron://[^/]+/", "")
  local path = target:match("^([^#]*)") or target

  local label = nil
  if display then
    display = display:gsub("^%s*(.-)%s*$", "%1")
    if display ~= "" then
      label = pandoc.Str(display)
    end
  else
    local title = note_title(path)
    -- a title that is only the mangled last path segment carries no information
    if title then
      local last = path:match("([^.]+)$") or path
      if slug(title) ~= slug(last) then
        label = pandoc.Emph({ pandoc.Str(title) })
      end
    end
  end

  if label then
    return {
      label,
      pandoc.Space(),
      pandoc.Str("("),
      path_node(target),
      pandoc.Str(")"),
    }
  end
  return { path_node(target) }
end

--- Text carried by an inline, or nil when the node is opaque to the scan.
local function text_of(inline)
  local t = inline.t
  if t == "Str" then
    return inline.text
  elseif t == "Space" or t == "SoftBreak" then
    return " "
  end
  return nil
end

function Inlines(inlines)
  -- cheap reject: nothing to do unless some Str actually holds "[["
  local seen = false
  for _, inline in ipairs(inlines) do
    if inline.t == "Str" and inline.text:find("[[", 1, true) then
      seen = true
      break
    end
  end
  if not seen then
    return nil
  end

  local out = {}
  local i = 1
  local n = #inlines
  while i <= n do
    local inline = inlines[i]
    local text = (inline.t == "Str") and inline.text or nil
    local open = text and text:find("[[", 1, true) or nil
    local consumed = i

    if open then
      -- accumulate forward over text-bearing inlines until "]]" closes the link
      local buffer = text:sub(open)
      local last = i
      local close = buffer:find("]]", 1, true)
      while not close and last < n do
        local nxt = text_of(inlines[last + 1])
        if not nxt then
          break
        end
        last = last + 1
        buffer = buffer .. nxt
        close = buffer:find("]]", 1, true)
      end
      if close then
        if open > 1 then
          table.insert(out, pandoc.Str(text:sub(1, open - 1)))
        end
        for _, node in ipairs(render(buffer:sub(3, close - 1))) do
          table.insert(out, node)
        end
        local tail = buffer:sub(close + 2)
        if tail ~= "" then
          table.insert(out, pandoc.Str(tail))
        end
        consumed = last
      else
        table.insert(out, inline)
      end
    else
      table.insert(out, inline)
    end
    i = consumed + 1
  end
  return out
end
