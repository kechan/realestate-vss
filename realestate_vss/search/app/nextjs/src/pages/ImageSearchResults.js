import React, { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import searchStyles from '../styles/SearchResults.module.css';
import imageSearchstyles from '../styles/ImageSearchResults.module.css';

export default function ImageSearchResults({ searchResults }) {
  const [modalVisible, setModalVisible] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [remarksModalVisible, setRemarksModalVisible] = useState(false);
  const [selectedRemarks, setSelectedRemarks] = useState(null);
  const modalRef = useRef();

  const openModal = (image) => {
    setSelectedImage(image);
    setModalVisible(true);
  };

  const closeModal = () => {
    setModalVisible(false);
  };

  const openRemarksModal = (remarks) => {
    setSelectedRemarks(remarks);
    setRemarksModalVisible(true);
  };

  const closeRemarksModal = () => {
    setRemarksModalVisible(false);
  };

  // const handleClickOutside = (event) => {
  //   if (modalRef.current && !modalRef.current.contains(event.target)) {
  //     closeModal();
  //   }
  // };

  // useEffect(() => {
  //   document.addEventListener("mousedown", handleClickOutside);
  //   return () => {
  //     document.removeEventListener("mousedown", handleClickOutside);
  //   };
  // });

  return (
    <div className={searchStyles['search-results']}>
      <div className={searchStyles['search-results-header']}>
        <h2>Listing</h2>
        <h2>Score</h2>
        <h2>Images</h2>
      </div>
      {searchResults.map(listing => {
        return (
          <div key={listing.listingId} className={searchStyles.listing}>
            <div className={searchStyles['listing-details']}>
              <div className={searchStyles['listing-id']}>
                <Link href={`/listing/${listing.listingId}`} passHref>
                  <span className={searchStyles['listing-link']}>{listing.listingId}</span>
                </Link>
              </div>
              <div className={searchStyles['listing-score']}>
                {listing.agg_score ? parseFloat(listing.agg_score).toFixed(2) : 'N/A'}
              </div>
              <div className={imageSearchstyles.images}>
                {listing.image_names.length > 0 ? (
                  listing.image_names.map(image_name => (
                    <img 
                      key={image_name} 
                      src={`${process.env.NEXT_PUBLIC_SEARCH_API_URL}/images/${image_name}`} 
                      alt={`Listing ${listing.listingId}`} 
                      onClick={() => openModal(`${process.env.NEXT_PUBLIC_SEARCH_API_URL}/images/${image_name}`)}
                    />
                  ))
                ) : (
                  <div className={imageSearchstyles['no-images']}>No images available</div>
                )}
              </div>
            </div>
            {listing.remarks ? (
              <div className={imageSearchstyles['listing-remarks']} onClick={() => openRemarksModal(listing.remarks)}>
                {listing.remarks.length > 100 ? listing.remarks.substring(0, 100) + '...' : listing.remarks}
              </div>
            ) : (
              <div className={imageSearchstyles['listing-remarks']}>
                Remarks not available
              </div>
            )}
          </div>
        );
      })}
      {modalVisible && (
        <div className={imageSearchstyles.modal} onClick={closeModal}>
          <div className={imageSearchstyles.modalWrapper} onClick={(e) => e.stopPropagation()}>
            <div className={imageSearchstyles.modalContent}>
              <img src={selectedImage} alt="Selected" className={imageSearchstyles.modalImage} />
              <span className={imageSearchstyles.close} onClick={closeModal}>&times;</span>
            </div>
          </div>
        </div>
      )}
      {remarksModalVisible && (
        <div className={imageSearchstyles.modal} onClick={closeRemarksModal}>
          <div className={imageSearchstyles.modalWrapper} onClick={(e) => e.stopPropagation()}>
            <div className={imageSearchstyles.modalContent}>
              <p className={imageSearchstyles.modalRemarks}>{selectedRemarks}</p>
              <span className={imageSearchstyles.close} onClick={closeRemarksModal}>&times;</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}