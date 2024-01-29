import React, { useState, useCallback } from 'react';
import styles from '../styles/FileUpload.module.css';

export default function FileUpload({ onFileChange, selectedFileUrl }) {
  const [selectedFile, setSelectedFile] = useState(null);
  // const [selectedFileUrl, setSelectedFileUrl] = useState(null);
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
    if (event.dataTransfer.files && event.dataTransfer.files[0]) {
      const file = event.dataTransfer.files[0];
      setSelectedFile(file);
      // setSelectedFileUrl(URL.createObjectURL(file));
      onFileChange(file);
    }
  }, [onFileChange]);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    // setSelectedFileUrl(URL.createObjectURL(file));
    onFileChange(file);
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
          onChange={handleFileChange}
          className={styles['file-input']}
          style={{ display: 'none' }}
        />
      {selectedFileUrl && <img src={selectedFileUrl} alt="Selected" className={styles.thumbnail} />}
    </div>
  );
}