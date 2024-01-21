import React, { useState } from 'react';
import styles from '../styles/FileUpload.module.css';

export default function FileUpload({ onFileChange }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedFileUrl, setSelectedFileUrl] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setSelectedFileUrl(URL.createObjectURL(event.target.files[0]));
    onFileChange(event.target.files[0]);
  };

  return (
    <div className={styles['file-upload-container']}>
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