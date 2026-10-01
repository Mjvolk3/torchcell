import React, {type ReactNode} from 'react';
import Link from '@docusaurus/Link';

type CardProps = {
  title: string;
  href: string;
  children?: ReactNode;
};

/** One outbound or internal link, shown as a card inside a LinkGrid. */
export function LinkCard({title, href, children}: CardProps): ReactNode {
  return (
    <Link className="tc-linkcard" to={href}>
      <span className="tc-linkcard__title">{title}</span>
      {children ? <span className="tc-linkcard__body">{children}</span> : null}
    </Link>
  );
}

export function LinkGrid({children}: {children: ReactNode}): ReactNode {
  return <div className="tc-linkgrid">{children}</div>;
}
