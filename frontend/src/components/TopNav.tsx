import React from 'react';
import { Search, Bell, Settings, User } from 'lucide-react';

const TopNav: React.FC = () => {
  return (
    <header className="top-nav">
      <div className="nav-tabs">
        <div className="nav-tab">Overview</div>
        <div className="nav-tab active">Classification</div>
        <div className="nav-tab">Datasets</div>
      </div>

      <div className="nav-actions">
        <div className="search-bar">
          <Search size={16} color="#908fa0" />
          <input type="text" placeholder="Search experiments..." />
        </div>
        <button className="icon-btn"><Bell size={20} /></button>
        <button className="icon-btn"><Settings size={20} /></button>
        <div className="user-avatar">
          <User size={18} />
        </div>
      </div>

      <style jsx>{`
        .top-nav {
          height: 64px;
          border-bottom: 1px solid var(--border-color);
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0 var(--spacing-md);
          background: var(--bg-deep);
        }

        .nav-tabs {
          display: flex;
          gap: var(--spacing-lg);
          height: 100%;
        }

        .nav-tab {
          height: 100%;
          display: flex;
          align-items: center;
          font-size: 14px;
          font-weight: 600;
          color: var(--text-secondary);
          cursor: pointer;
          position: relative;
          padding: 0 var(--spacing-xs);
        }

        .nav-tab.active {
          color: var(--text-primary);
        }

        .nav-tab.active::after {
          content: '';
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          height: 2px;
          background: var(--primary);
        }

        .nav-actions {
          display: flex;
          align-items: center;
          gap: var(--spacing-md);
        }

        .search-bar {
          background: var(--bg-surface);
          border: 1px solid var(--border-color);
          border-radius: var(--radius-sm);
          padding: 6px 12px;
          display: flex;
          align-items: center;
          gap: var(--spacing-xs);
          width: 280px;
        }

        .search-bar input {
          background: transparent;
          border: none;
          color: var(--text-primary);
          font-size: 13px;
          width: 100%;
          outline: none;
        }

        .icon-btn {
          background: transparent;
          color: var(--text-secondary);
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 4px;
        }

        .icon-btn:hover {
          color: var(--text-primary);
        }

        .user-avatar {
          width: 32px;
          height: 32px;
          background: var(--bg-surface-highest);
          border-radius: var(--radius-full);
          display: flex;
          align-items: center;
          justify-content: center;
          border: 1px solid var(--border-color);
          color: var(--primary);
        }
      `}</style>
    </header>
  );
};

export default TopNav;
