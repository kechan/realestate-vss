// TextSearch.js
import React from 'react';
import styles from '../styles/TextSearch.module.css';

export default function TextSearch({ searchTerm, onSearchTermChange, onSearchSubmit }) {
  // This function is triggered when the user presses a key while focused on the input.
  const handleKeyDown = (e) => {
    // Check if the key pressed is "Enter"
    if (e.key === 'Enter') {
      // If it is, call the onSearchSubmit function passed via props.
      // onSearchSubmit(e.target.value);
      e.preventDefault();
      onSearchSubmit(e);
    }
  };

  return (
    <div className={styles['text-search-container']}>
      <input
        type="text"
        placeholder="Enter search terms..."
        value={searchTerm}
        // Call onSearchTermChange every time the input changes, to update the searchTerm state in the 
        // parent component.
        onChange={(e) => onSearchTermChange(e.target.value)}
        // Call handleKeyDown when any key is pressed.
        onKeyDown={handleKeyDown}
        className={styles['text-input']}
      />
    </div>
  );
}
