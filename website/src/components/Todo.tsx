import React, {type ReactNode} from 'react';

type Props = {
  /** What is missing and where it must come from. */
  children?: ReactNode;
};

/**
 * A visibly marked placeholder for a value that has not been sourced yet.
 * Renders as `TODO(source from SI)` unless other text is given. Use it instead of
 * guessing an experimental detail.
 */
export default function Todo({children}: Props): ReactNode {
  return <span className="tc-chip tc-chip--todo">TODO({children ?? 'source from SI'})</span>;
}
