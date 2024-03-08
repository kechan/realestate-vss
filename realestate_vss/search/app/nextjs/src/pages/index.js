import React, { useState, useRef, useEffect } from 'react';
import Tooltip from '@material-ui/core/Tooltip';

import TextSearch from './TextSearch';
import FileUpload from './FileUpload';
import CriteriaSearchForm from './CriteriaSearchForm';

import ImageSearchResults from './ImageSearchResults'; 
import TextSearchResults from './TextSearchResults';

import { ClearCacheButton } from './Banner';


export default function Home({ bannerHeight}) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedFileUrl, setSelectedFileUrl] = useState(null);

  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState([]);

  const [criteriaSearchFormData, setCriteriaSearchFormData] = useState({});

  const [textSearchMode, setTextSearchMode] = useState('VSS_ONLY');

  // Load the search results from Local Storage when the component mounts
  useEffect(() => {
    // console.log('first effect')
    const savedSearchResults = localStorage.getItem('searchResults');
    const savedSearchTerm = localStorage.getItem('searchTerm');
    const savedTextSearchMode = localStorage.getItem('textSearchMode');

    if (savedSearchResults) {
      const results = JSON.parse(savedSearchResults);
      const searchType = localStorage.getItem('searchType');
      if (searchType) {
        results.searchType = searchType;
      }
      setSearchResults(results);
    }

    if (savedSearchTerm) {
      setSearchTerm(savedSearchTerm);
    }

    if (savedTextSearchMode) {
      setTextSearchMode(savedTextSearchMode);
    }
    
  }, []); // Empty dependency array ensures this runs once on client-side mount
  

  // Save the search results to Local Storage whenever they change
  useEffect(() => {
    if (searchResults.length > 0) {
      localStorage.setItem('searchResults', JSON.stringify(searchResults));
      localStorage.setItem('searchType', searchResults.searchType); // Save the search type separately
    }
  }, [searchResults]); // Runs this effect whenever searchResults changes

  // useEffect(() => {
  //   // Assuming textSearchMode is a state variable in your component
  //   localStorage.setItem('textSearchMode', textSearchMode);
  //   console.log('Saving textSearchMode to localStorage:', textSearchMode);
  // }, [textSearchMode]);


  const handleFileChange = (file) => {
    setSelectedFile(file);
    // Clear the searchTerm when a file is selected for upload
    if (searchTerm) setSearchTerm('');
    setSelectedFileUrl(file ? URL.createObjectURL(file) : null);

    // Clear the criteria search
    if (!isFormDataEmpty(criteriaSearchFormData)) {
      setCriteriaSearchFormData({});
    }
  }

  const handleSearchTermChange = (newSearchTerm) => {
    setSearchTerm(newSearchTerm);
    localStorage.setItem('searchTerm', newSearchTerm);   // persists to work for browser back button
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

  const handleCriteriaFormChange = (updatedFormData) => {
    // console.log('Entering handleCriteriaFormChange')
    // console.log('Updated form data:', updatedFormData);

    if (selectedFile) {
      setSelectedFile(null);
      setSelectedFileUrl(null);
    }

    // its ok to include search term since this is both VSS + criteria search
    // if (searchTerm) {
    //   setSearchTerm('');
    // }
  };

  const isFormDataEmpty = (formData) => {
    if (formData === null) {
      return true;
    }
  
    for (let key in formData) {
      if (Array.isArray(formData[key])) {
        if (formData[key].some(value => value !== null)) {
          return false;
        }
      } else if (formData[key] !== null && formData[key] !== "") {
        return false;
      }
    }
    return true;
  };

  // Function to handle calling search API 
  const handleSubmit = async (event) => {
    console.log('search mode:', textSearchMode);
    console.log('search term:', searchTerm); 
    event.preventDefault();
    // setSearchResults([]); // don't clear this here, otherwise the UX will flicker a lot

    if (!selectedFile && !searchTerm && isFormDataEmpty(criteriaSearchFormData)) {
      alert('Please select a file, enter a search term, or fill out the criteria search form.');
      return;
    }

    console.log('Search submitted:', criteriaSearchFormData);

    const apiURL = process.env.NEXT_PUBLIC_SEARCH_API_URL;
    // console.log('apiURL:', apiURL);

    if (selectedFile) {   // perform image (to image) search
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
      return
    }

    if (textSearchMode === 'VSS_ONLY') {

      if (searchTerm) {
        if (criteriaSearchFormData.provState == "" || criteriaSearchFormData.provState == null) {
          alert('Please select a province.');
          return;
        }
        const input_payload = {
          provState: criteriaSearchFormData.provState,    // TODO: hard-coded for now, need to change later
          phrase: searchTerm
        }

        try {
          const response = await fetch(`${apiURL}/search-by-text/`, {   // VSS_ONLY by default
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
      else {
        alert('Please enter a search term.');
      }

      return;
    }

    if (textSearchMode === 'SOFT_MATCH_AND_VSS') {
      console.log('Performing soft match + VSS search');
      if (criteriaSearchFormData.provState == "") {
        alert('Please select a province.');
        return;
      }

      try {
        criteriaSearchFormData.phrase = searchTerm;
        const response = await fetch(`${apiURL}/search-by-text/?mode=SOFT_MATCH_AND_VSS&lambda_val=0.8&alpha_val=0.5`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(criteriaSearchFormData)
        });
        const data = await response.json();

        data.searchType = 'text'; // Add a property to the data object to indicate the search type
        setSearchResults(data); // Update the state with the search results

      } catch (error) {
        console.error('Error:', error);
      }
      return
    }

    if (textSearchMode === 'TEXT_TO_IMAGE_VSS') {
      if (!searchTerm || searchTerm === "") {
        alert('Please enter a search term.');
        return;
      }
      const input_payload = {
        phrase: searchTerm
      }
      try {
        const response = await fetch(`${apiURL}/text-to-image-search/`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(input_payload)
        });
        const data = await response.json();

        data.searchType = 'image'; // Add a property to the data object to indicate the search type
        setSearchResults(data); // Update the state with the search results
      } catch (error) {
        console.error('Error:', error);        
      }

      return;
    }

    // we reach here if non of above conditions are met (this explicitly requires all above have a return)
    alert(`Invalid search: selectedFile=${selectedFile}, selectedFileUrl=${selectedFileUrl} textSearchMode=${textSearchMode}, searchTerm=${searchTerm}, criteriaSearchFormData=${JSON.stringify(criteriaSearchFormData)}`);

    // TODO: if we get here, we should consider clear the previous search results and render an error
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
        
          <div className="text-search-model-select-container">
            <Tooltip title="Choose a search mode: 'VSS ONLY' for only Vector Similarity Search, 'Soft Match + VSS' for a mix of criteria matching and VSS." placement="top">
              <div>

                {/* <label for="text-search-mode-select">Mode:</label> */}
                <select className="text-search-mode-select" 
                        style={{ color: '#FFFFFF' }} 
                        value={textSearchMode} 
                        // onChange={e => setTextSearchMode(e.target.value)}
                        onChange={e => {
                          const newMode = e.target.value;
                          setTextSearchMode(newMode); // Update state
                          localStorage.setItem('textSearchMode', newMode); // Save to localStorage
                          console.log('Text search mode updated to:', newMode); // Optional: for debugging
                        }}
                >
                  <option value="VSS_ONLY">🔍 by Remarks only </option>
                  <option value="SOFT_MATCH_AND_VSS">🔍📃 by Remarks and Criteria</option>
                  <option value="TEXT_TO_IMAGE_VSS">📝➡️🖼️ Text to Image</option>
                </select>

              </div>
            </Tooltip>
          </div>

          <ClearCacheButton />
          
        </div>      

        <CriteriaSearchForm
            setSearchCriteria={setCriteriaSearchFormData}
            onFormChange={handleCriteriaFormChange}
            searchMode={textSearchMode}
          />
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
        max-width: 800px; /* Adjust width as needed */
        display: flex;
        flex-direction: column;
        align-items: center;
      }

      .search-container-tighter {
        border-radius: 24px;
        box-shadow: 0px 4px 6px rgba(32, 33, 36, 0.28);
        padding: 15px;
        margin: 10px auto;
        max-width: 800px; /* Adjust width as needed */
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
        width: 90%;
      }

      .text-search-model-select-container {
        display: flex;
        flex-direction: column;
        // align-items: center;
        gap: 5px;
      }

      .text-search-model-select-container label {
        font-size: 14px;
        color: #000000;
      }

      .text-search-mode-select {
        padding: 5px 5px;
        font-size: 10px;
        border-radius: 5px;
        border: 1px solid #D2232A;
        background-color: #D2232A;
        appearance: none;
      }
    `}</style>

    
    </div>

    </>
  );
}
