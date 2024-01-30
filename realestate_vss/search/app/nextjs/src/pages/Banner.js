import React from 'react';
import styles from '../styles/Banner.module.css'; // Assuming a CSS module for styling the banner

const Banner = React.forwardRef((props, ref) => {
  return (
    <div className={styles.banner} ref={ref}>
      <svg className={styles.logo} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        {/* SVG content can be placed here */}
        <circle cx="12" cy="12" r="10" stroke="white" strokeWidth="2" fill="white" />
      </svg>
      <h1 className={styles.appName}>Listing Image or Semantic Search</h1>
    </div>
  );
});

export default Banner;
