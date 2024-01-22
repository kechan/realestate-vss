import React, { useState, useRef, useEffect } from 'react';
import FileUpload from './FileUpload';
import SearchResults from './SearchResults'; 

export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [searchResults, setSearchResults] = useState([]);

  const bannerRef = useRef(null);
  const [bannerHeight, setBannerHeight] = useState(0);

  useEffect(() => {
    setBannerHeight(bannerRef.current.offsetHeight);
  }, []);

  const handleFileChange = (file) => {
    setSelectedFile(file);
  }

  // Function to handle form submission for file upload
  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      alert('Please select a file first!');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/search-by-image/', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();

      setSearchResults(data); // Update the state with the search results
    } catch (error) {
      console.error('Error:', error);
    }
  };

  // Determine the class for the search-container based on presence or absence of search results
  const searchContainerClass = searchResults.length > 0 ? "search-container-tighter" : "search-container";

  return (
    <>
    <div className="banner" ref={bannerRef}>
      <svg className="logo" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <circle cx="12" cy="12" r="10" stroke="white" strokeWidth="2" fill="white" />
      </svg>
      <h1 className="app-name">Listing Image Search</h1>

      <style jsx>{`
        .banner {
          display: flex;
          align-items: center;
          width: 100%;
          padding: 20px;
          background-color: #D2232A;
          color: white;
          position: fixed; // Fix the banner to the top
          // position: sticky; 
          top: 0; // Align the top of the banner with the top of the viewport
          left: 0; // Align the left of the banner with the left of the viewport
          z-index: 1000; // Ensure the banner is above other content
        }
        .logo {
          width: 50px;
          height: 50px;
          margin-right: 20px;
        }
        .app-name {
          // font-family: 'Exo 2', sans-serif;
          font-family: 'Roboto', sans-serif;
          color: #fff; 
        }
      `}</style>
    </div>

    <div className="container" style={{ paddingTop: bannerHeight + 10}}>
      <div className={searchContainerClass}>
        <FileUpload onFileChange={handleFileChange} />      
        <button className="search-btn" onClick={handleSubmit}>Search</button>      
      </div>

    {/* Display search results */}
    {searchResults.length > 0 && <SearchResults searchResults={searchResults} />}

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
        margin-top: 10px; // Add some space between the buttons
      }

      .search-btn:hover {
        background-color: #A81B22;
      }
    `}</style>

    
    </div>

    </>
  );
}
