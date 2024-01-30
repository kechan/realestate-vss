import React, { useState, useRef, useEffect } from 'react';
import TextSearch from './TextSearch';
import FileUpload from './FileUpload';
import ImageSearchResults from './ImageSearchResults'; 
import TextSearchResults from './TextSearchResults';

export default function Home({ bannerHeight}) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedFileUrl, setSelectedFileUrl] = useState(null);

  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState([]);

  const handleFileChange = (file) => {
    setSelectedFile(file);
    // Clear the searchTerm when a file is selected for upload
    if (searchTerm) setSearchTerm('');
    setSelectedFileUrl(file ? URL.createObjectURL(file) : null);
  }

  const handleSearchTermChange = (newSearchTerm) => {
    setSearchTerm(newSearchTerm);
    // Clear the selectedFile when a searchTerm is entered
    if (selectedFile) {
      setSelectedFile(null);
      setSelectedFileUrl(null);
    }
  };

  const handleTextSearchSubmit = async (event) => {
    // console.log('Search submitted:', searchTerm);
    event.preventDefault();
    // setSearchResults([]); // don't clear this yet, otherwise the UX will flicker a lot
    handleSubmit(event);
  };

  // Function to handle form submission for file upload
  const handleSubmit = async (event) => {
    event.preventDefault();
    // setSearchResults([]); // don't clear this yet, otherwise the UX will flicker a lot
    if (!selectedFile && !searchTerm) {
      alert('Please select a file or enter a search term.');
      return;
    }

    const apiURL = process.env.NEXT_PUBLIC_SEARCH_API_URL;
    console.log('apiURL:', apiURL);

    if (selectedFile) {
      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const response = await fetch(`${apiURL}/search-by-image/`, {
          method: 'POST',
          body: formData,
        });
        const data = await response.json();

        data.searchType = 'image'; // Add a property to the data object to indicate the search type
        setSearchResults(data); // Update the state with the search results
      } catch (error) {
        console.error('Error:', error);
      }
    }

    if (searchTerm) {
      // implement search by text
      const input_payload = {
        provState: "ON",    // TODO: hard-coded for now, need to change later
        phrase: searchTerm
      }

      try {
        const response = await fetch(`${apiURL}/search-by-text/`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(input_payload)
        });
        const data = await response.json();

        // console.log('data:', data);
        data.searchType = 'text'; // Add a property to the data object to indicate the search type
        setSearchResults(data); // Update the state with the search results
      } catch (error) {
        console.error('Error:', error);
      }
    }
  };

  // Determine the class for the search-container based on presence or absence of search results
  const searchContainerClass = searchResults.length > 0 ? "search-container-tighter" : "search-container";

  return (
    <>

    <div className="container" style={{ paddingTop: bannerHeight + 10}}>
      <div className={searchContainerClass}>
        <FileUpload onFileChange={handleFileChange} selectedFileUrl={selectedFileUrl}/>
        <div className="text-and-button">
          <TextSearch 
            searchTerm={searchTerm}
            onSearchTermChange={handleSearchTermChange}
            onSearchSubmit={handleTextSearchSubmit}
          />
          <button className="search-btn" onClick={handleSubmit}>Search</button>
        </div>      
      </div>

    {/* Display search results */}
    {searchResults.length > 0 && searchResults.searchType === 'image' && <ImageSearchResults searchResults={searchResults} />}
    {searchResults.length > 0 && searchResults.searchType === 'text' && <TextSearchResults searchResults={searchResults} />}

    <style jsx>{`
   
      .container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        background-color: #f7f7f7;
        // padding-top: 100px; // Add a top padding
      }

      .search-container {
        background: #fff;
        border-radius: 24px;
        box-shadow: 0px 4px 6px rgba(32, 33, 36, 0.28);
        padding: 20px;
        margin: 20px auto;
        max-width: 600px; /* Adjust width as needed */
        display: flex;
        flex-direction: column;
        align-items: center;
      }

      .search-container-tighter {
        border-radius: 24px;
        box-shadow: 0px 4px 6px rgba(32, 33, 36, 0.28);
        padding: 15px;
        margin: 10px auto;
        max-width: 600px; /* Adjust width as needed */
        display: flex;
        flex-direction: column;
        align-items: center;
      }

      .search-btn {
        border: none;
        // background-color: #28a745;
        background-color: #D2232A;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        // margin-top: 10px; // Add some space between the buttons
      }

      .search-btn:hover {
        background-color: #A81B22;
      }

      .text-and-button {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 20px; /* Adjust as needed for spacing */
        width: 80%;
      }
    `}</style>

    
    </div>

    </>
  );
}
