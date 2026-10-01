import React, {type ReactNode} from 'react';
import clsx from 'clsx';

export type BadgeStatus =
  | 'planned'
  | 'provisional'
  | 'verified'
  | 'rejected'
  | 'withdrawn'
  | 'baseline';

const TITLES: Record<BadgeStatus, string> = {
  planned: 'Not built yet',
  provisional: 'Graded by the server, not yet reproduced',
  verified: "Reproduced from the submitter's code",
  rejected: 'Failed validation; no score was recorded',
  withdrawn: 'Withdrawn after grading; not shown on a board',
  baseline: 'Standard baseline run by the TorchCell maintainers',
};

type Props = {
  status: BadgeStatus;
  /** Overrides the visible label; the default is the status word itself. */
  label?: string;
};

/** A text chip for a status. The word carries the meaning; color only reinforces it. */
export default function StatusBadge({status, label}: Props): ReactNode {
  return (
    <span className={clsx('tc-chip', `tc-chip--${status}`)} title={TITLES[status]}>
      {label ?? status}
    </span>
  );
}
