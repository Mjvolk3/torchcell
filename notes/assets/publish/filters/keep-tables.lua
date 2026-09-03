--- notes/assets/publish/filters/keep-tables.lua
---
--- Keep a table and its caption on one page, but only when the table can fit on one.
---
--- Pandoc emits every markdown table as a `longtable`, whose inter-row breakpoints are
--- free. A short table that starts low on a page therefore splits instead of moving down
--- whole, because TeX scores an underfull page as infinitely bad and so always prefers to
--- fill. Only an infinite penalty between rows moves the table, and an infinite penalty is
--- also how a table LONGER than a page gets silently truncated: the overflow falls off the
--- bottom and the build still exits 0.
---
--- So the decision has to be made where the table's size is known, which is here. The
--- header (`assets/publish/tex-templates/header-includes.tex`) defines the switch;
--- this filter throws it around each table small enough that keeping it whole is safe.
---
--- The size test is deliberately two-sided, because either one alone lets a page-tall
--- table through: a table can be tall through many short rows, or through few rows whose
--- cells each wrap to a dozen lines. The dataset-triage note is the second kind, roughly
--- one printed page PER ROW, and a row-count rule alone dropped six thousand words of it.
--- Both limits are set well under a page at \footnotesize.

local stringify = pandoc.utils.stringify

local MAX_ROWS = 20
local MAX_CHARS = 2500

local function row_count(tbl)
  local n = 0
  for _, body in ipairs(tbl.bodies or {}) do
    n = n + #(body.body or {})
    n = n + #(body.head or {})
  end
  if tbl.head and tbl.head.rows then
    n = n + #tbl.head.rows
  end
  if tbl.foot and tbl.foot.rows then
    n = n + #tbl.foot.rows
  end
  return n
end

function Table(tbl)
  if not (FORMAT and FORMAT:match("latex")) then
    return nil
  end
  local rows = row_count(tbl)
  local chars = #stringify(tbl)
  if rows == 0 or rows > MAX_ROWS or chars > MAX_CHARS then
    return nil
  end
  return {
    pandoc.RawBlock("latex", "\\tcltkeepon"),
    tbl,
    pandoc.RawBlock("latex", "\\tcltkeepoff"),
  }
end
