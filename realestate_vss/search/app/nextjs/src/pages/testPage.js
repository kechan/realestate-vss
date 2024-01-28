// testPage.js
import React, { useState } from 'react';
import TextSearch from './TextSearch';

const TestPage = () => {
  const [searchTerm, setSearchTerm] = useState('');

  const handleSearchTermChange = (newTerm) => {
    setSearchTerm(newTerm);
  };

  const handleSearchSubmit = (term) => {
    // This is where you would likely make a call to your backend to perform the search.
    console.log('Search submitted for term:', term);
  };

  return (
    <TextSearch 
      searchTerm={searchTerm}
      onSearchTermChange={handleSearchTermChange}
      onSearchSubmit={handleSearchSubmit}
    />
  );
};

export default TestPage;
