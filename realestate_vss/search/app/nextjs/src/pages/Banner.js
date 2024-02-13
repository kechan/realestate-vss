import React from 'react';
import styles from '../styles/Banner.module.css'; // Assuming a CSS module for styling the banner

const Banner = React.forwardRef((props, ref) => {
  return (
    <div className={styles.banner} ref={ref}>
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
      <h1 className={styles.appName}>Listing Image or Remarks Search</h1>
    </div>
  );
});

export default Banner;
