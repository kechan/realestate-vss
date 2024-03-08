import React from 'react';
import styles from '../styles/Banner.module.css'; // Assuming a CSS module for styling the banner

export const ClearCacheButton = () => {
  const clearCache = () => {
    localStorage.clear();
    alert('Cache cleared!');
  }

  return (
    <button onClick={clearCache} style={{
      position: 'absolute', 
      bottom: '10px', 
      right: '10px',
      color: 'black', // Ensuring text color is black for visibility
      backgroundColor: 'white', // Optional: if you want to add a background color
      padding: '10px', // Optional: if you want to add some padding
      borderRadius: '5px', // Optional: if you want to round the corners
      border: 'none', // Optional: if you want to remove the default button border
      cursor: 'pointer', // Optional: to show a pointer on hover
    }}>
      Clear Cache
    </button>
  );
}

const Banner = React.forwardRef((props, ref) => {
  return (
    <div className={styles.banner} ref={ref}>
      <div className={styles.header}>
        <svg className={styles.logo} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          {/* <circle cx="12" cy="12" r="10" stroke="white" strokeWidth="2" fill="white" /> */}
          {/* Magnifying glass */}
          <circle cx="11" cy="11" r="7" stroke="white" strokeWidth="2" />
          <line x1="21" y1="21" x2="15.65" y2="15.65" stroke="white" strokeWidth="2" />

          <polygon points="8,10 11,7 14,10" fill="white" />
          <rect x="9" y="10" width="4" height="4" fill="white" />
          {/* Door */}
          <rect x="10.5" y="12" width="1" height="2" fill="currentColor" />
        </svg>
        <h1 className={styles.appName}>Multi-Cross-Modal Listing Search</h1>
      </div>
      <h2 className={styles.modalities}>Descriptive Text and Image</h2>
      <h2 className={styles.modalities}>Searching over 1.4 million images (from ~48k listings) and 0.25 million remarks.</h2>
    </div>
  );
});

export default Banner;
