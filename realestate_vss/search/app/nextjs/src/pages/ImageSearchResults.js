import React from 'react';
import styles from '../styles/ImageSearchResults.module.css';

export default function ImageSearchResults({ searchResults }) {
  return (
    <div className={styles['search-results']}>
      <div className={styles['search-results-header']}>
        <h2>Listing</h2>
        <h2>Score</h2>
        <h2>Images</h2>
      </div>
      {searchResults.map(listing => (
        <div key={listing.listingId} className={styles.listing}>
          <div className={styles['listing-id']}>{listing.listingId}</div>
          <div className={styles['listing-score']}>
            {listing.avg_score ? parseFloat(listing.avg_score).toFixed(2) : 'N/A'}
          </div>
          <div className={styles.images}>
            {listing.image_names.map(image_name => (
              <img key={image_name} src={`http://localhost:8000/images/${image_name}`} alt={`Listing ${listing.listingId}`} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}