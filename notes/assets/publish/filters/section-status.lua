--[[
  section-status.lua -- provenance chips on note headings.

  Marks every section of a note PDF with whether a human has read it, using the same
  three-symbol visual language as paper/nature-biotech/editing.pdf so a note and the
  manuscript read as one document family. The MEANING differs, because the question a
  note has to answer is not "is this publication ready" but "did a person check this":

      ai    red X       written by an agent, NOT yet read by the author
      tent  amber sq    read, provisional, expected to change
      read  green check read and checked by the author

  Usage in markdown, as a pandoc header attribute, which stays invisible in Dendron and
  in any markdown viewer:

      ## 2026.09.03 - Something the author verified {.read}
      ### A subsection still under review {.tent}

  THE DEFAULT IS `ai`, AND THAT IS THE POINT. An unmarked heading gets the red chip
  rather than no chip. A blank heading would be indistinguishable from a checked one,
  which is exactly the confusion these marks exist to remove: a document can never
  silently imply review it has not had. Marking a section `read` is a positive act.
--]]

local STATUS = { ai = true, tent = true, read = true }
local DEFAULT = "ai"

--- Pull a status class off a heading, returning it and the surviving classes.
local function take_status(classes)
  local found, keep = nil, {}
  for _, c in ipairs(classes) do
    if STATUS[c] and not found then
      found = c
    else
      keep[#keep + 1] = c
    end
  end
  return found, keep
end

function Header(el)
  -- Only the levels a reader navigates by. Deeper headings would clutter the page and
  -- are covered by their parent's mark.
  if el.level > 3 then
    return nil
  end
  local status, keep = take_status(el.classes)
  el.classes = keep
  el.content:insert(pandoc.RawInline("latex", "\\secstatus{" .. (status or DEFAULT) .. "}"))
  return el
end

--- Prepend the legend so the symbols are defined for the reader before they meet one.
-- Body blocks render after the table of contents, which is where this belongs: the
-- contents page carries the chips too, since they sit inside the heading text.
function Pandoc(doc)
  doc.blocks:insert(1, pandoc.RawBlock("latex", "\\notestatuslegend"))
  return doc
end
