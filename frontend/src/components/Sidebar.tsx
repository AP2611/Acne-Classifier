import React from 'react';
import { LayoutDashboard, ScanSearch, Library, ScrollText, PlusCircle } from 'lucide-react';

const Sidebar: React.FC = () => {
  const menuItems = [
    { icon: <LayoutDashboard size={20} />, label: 'Dashboard', active: false },
    { icon: <ScanSearch size={20} />, label: 'Classification', active: true },
    { icon: <Library size={20} />, label: 'Model Library', active: false },
    { icon: <ScrollText size={20} />, label: 'Training Logs', active: false },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-logo">N</div>
        <div className="brand-name">NEXUS ML</div>
      </div>

      <div className="sidebar-project">
        <div className="project-info">
          <span className="project-name">Project Alpha</span>
          <span className="project-version">Production v2.4</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        {menuItems.map((item, idx) => (
          <div key={idx} className={`nav-item ${item.active ? 'active' : ''}`}>
            {item.icon}
            <span className="nav-label">{item.label}</span>
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <button className="new-experiment-btn">
          <PlusCircle size={18} />
          <span>New Experiment</span>
        </button>
      </div>

      <style jsx>{`
        .sidebar {
          width: 260px;
          background: var(--bg-deep);
          border-right: 1px solid var(--border-color);
          display: flex;
          flex-direction: column;
          padding: var(--spacing-md);
        }

        .sidebar-brand {
          display: flex;
          align-items: center;
          gap: var(--spacing-sm);
          margin-bottom: var(--spacing-xl);
          padding: 0 var(--spacing-xs);
        }

        .brand-logo {
          width: 32px;
          height: 32px;
          background: var(--primary);
          color: white;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: 800;
          border-radius: var(--radius-sm);
        }

        .brand-name {
          font-size: 18px;
          font-weight: 700;
          letter-spacing: 0.05em;
        }

        .sidebar-project {
          padding: var(--spacing-sm);
          background: var(--bg-surface-high);
          border-radius: var(--radius-md);
          margin-bottom: var(--spacing-lg);
          border: 1px solid var(--border-color);
        }

        .project-name {
          display: block;
          font-weight: 600;
          font-size: 14px;
        }

        .project-version {
          font-size: 11px;
          color: var(--text-muted);
          font-family: var(--font-mono);
        }

        .sidebar-nav {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: var(--spacing-xs);
        }

        .nav-item {
          display: flex;
          align-items: center;
          gap: var(--spacing-sm);
          padding: var(--spacing-sm);
          border-radius: var(--radius-sm);
          color: var(--text-secondary);
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .nav-item:hover {
          background: var(--bg-surface);
          color: var(--text-primary);
        }

        .nav-item.active {
          background: rgba(99, 102, 241, 0.1);
          color: var(--primary);
          border-left: 3px solid var(--primary);
          padding-left: calc(var(--spacing-sm) - 3px);
        }

        .sidebar-footer {
          margin-top: auto;
        }

        .new-experiment-btn {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: var(--spacing-xs);
          padding: var(--spacing-sm);
          background: var(--primary);
          color: white;
          border-radius: var(--radius-sm);
          font-weight: 600;
          font-size: 14px;
        }

        .new-experiment-btn:hover {
          opacity: 0.9;
          transform: translateY(-1px);
          box-shadow: 0 4px 12px var(--primary-glow);
        }
      `}</style>
    </aside>
  );
};

export default Sidebar;
