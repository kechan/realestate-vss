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
      <div className={styles['btn-container']}>
        {/* <button className={styles.btn}>Choose File</button> */}
        <label className={styles['file-input-presentation']}>Upload an image</label>
        <input type="file" name="myfile" onChange={handleFileChange} className={styles['file-input']} />
      </div>
      {selectedFileUrl && <img src={selectedFileUrl} alt="Selected" className={styles.thumbnail} />}
    </div>
  );
}