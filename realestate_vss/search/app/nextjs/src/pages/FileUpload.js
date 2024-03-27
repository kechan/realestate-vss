import React, { useState, useCallback } from 'react';
import styles from '../styles/FileUpload.module.css';

export default function FileUpload({ onFileChange, selectedFileUrls }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [dragOver, setDragOver] = useState(false);

  const handleDragOver = useCallback((event) => {
    event.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((event) => {
    event.preventDefault();
    setDragOver(false);
  }, []);

  const handleDrop = useCallback((event) => {
    event.preventDefault();
    event.stopPropagation();
    setDragOver(false);
    console.log(event.dataTransfer.files);
    if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {      

      // const files = Array.from(event.dataTransfer.files);
      // setSelectedFiles(files);
      // onFileChange(files);

      const newFiles = Array.from(event.dataTransfer.files);
      setSelectedFiles(prevFiles => [...prevFiles, ...newFiles]);
      onFileChange([...selectedFiles, ...newFiles]);
      
    }
  }, [onFileChange]);

  const handleFileChange = (event) => {
    // const files = Array.from(event.target.files);
    // setSelectedFiles(files);
    // onFileChange(files);
    const newFiles = Array.from(event.target.files);
    setSelectedFiles(prevFiles => [...prevFiles, ...newFiles]);
    onFileChange([...selectedFiles, ...newFiles]);
    event.target.value = null; // Clear the file input
  };

  const handleRemove = (index) => {
    const newFiles = [...selectedFiles];
    newFiles.splice(index, 1);
    setSelectedFiles(newFiles);
    onFileChange(newFiles);
  };

  return (
    <div 
      className={`${styles['file-upload-container']} ${dragOver ? styles['drag-over'] : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
        <span className={styles['drag-text']}>Drag an image here or</span>
        <label htmlFor="file-upload" className={styles['upload-link']}>
          upload a file
        </label>
        <input
          id="file-upload"
          type="file"
          multiple
          onChange={handleFileChange}
          className={styles['file-input']}
          style={{ display: 'none' }}
        />
      {/* {selectedFileUrls && selectedFileUrls.map((url, index) => (
        <img key={index} src={url} alt="Selected" className={styles.thumbnail} />
      ))} */}
      {selectedFileUrls && selectedFileUrls.map((url, index) => (
        <div key={index} className={styles.thumbnailContainer}>
          <button onClick={() => handleRemove(index)} className={styles.removeButton}>Remove</button>
          <img src={url} alt="Selected" className={styles.thumbnail} />
          
        </div>
      ))}
    </div>
  );
}