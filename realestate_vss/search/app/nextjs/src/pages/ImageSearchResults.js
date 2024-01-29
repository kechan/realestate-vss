import React from 'react';
import searchStyles from '../styles/SearchResults.module.css';
import imageSearchstyles from '../styles/ImageSearchResults.module.css';

export default function ImageSearchResults({ searchResults }) {
  return (
    <div className={searchStyles['search-results']}>
      <div className={searchStyles['search-results-header']}>
        <h2>Listing</h2>
        <h2>Score</h2>
        <h2>Images</h2>
      </div>
      {searchResults.map(listing => (
        <div key={listing.listingId} className={searchStyles.listing}>
          <div className={searchStyles['listing-id']}>{listing.listingId}</div>
          <div className={searchStyles['listing-score']}>
            {listing.avg_score ? parseFloat(listing.avg_score).toFixed(2) : 'N/A'}
          </div>
          <div className={imageSearchstyles.images}>
            {listing.image_names.map(image_name => (
              <img key={image_name} src={`http://localhost:8000/images/${image_name}`} alt={`Listing ${listing.listingId}`} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}