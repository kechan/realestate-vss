import { useRouter } from 'next/router';
import { useEffect, useState, useRef } from 'react';
import styles from '../../styles/[listingId].module.css';


function PropertyIdentificationCard({ jumpId, streetName, city, provState, postalCode }) {
  return (
    <div className={styles.PropertyDetailsCard}>
      <h2>Property Identification</h2>
      <p><strong>Listing ID:</strong> {jumpId}</p>
      <p><strong>Address:</strong> {`${streetName}, ${city}, ${provState}, ${postalCode}`}</p>
    </div>
  );
}

function PropertyDetailsCard({ propertyType, transactionType, price, leasePrice, beds, baths, sizeInterior, sizeInteriorUOM, lotSize, lotUOM, carriageTrade}) {
  const displayPrice = transactionType === 'LEASE' ? leasePrice : price;

  return (
    <div className={styles.PropertyDetailsCard}>
      <h2>Property Details</h2>
      <p><strong>Type:</strong> {propertyType}</p>
      <p><strong>Transaction:</strong> {transactionType}</p>
      <p><strong>Price:</strong> {displayPrice.toLocaleString('en-US', { style: 'currency', currency: 'CAD' })}</p>
      <p><strong>Beds:</strong> {beds}</p>
      <p><strong>Baths:</strong> {baths}</p>
      {sizeInterior && <p><strong>Interior Size:</strong> {`${sizeInterior} ${sizeInteriorUOM}`}</p>}
      {lotSize && <p><strong>Lot Size:</strong> {`${lotSize} ${lotUOM}`}</p>}
      {carriageTrade && <p><strong>Carriage Trade:</strong> {carriageTrade}</p>}
    </div>
  );
}

// function PropertyFeaturesCard({ features }) {
//   return (
//     <div className={styles.PropertyDetailsCard}>
//       <h2>Property Features</h2>
//       <ul>
//         {features.parking && <li>Parking</li>}
//         {features.pool && <li>Pool</li>}
//         {features.garage && <li>Garage</li>}
//         {features.waterFront && <li>Waterfront</li>}
//         {features.fireplace && <li>Fireplace</li>}
//         {features.ac && <li>Air Conditioning</li>}
//       </ul>
//     </div>
//   );
// }

function PropertyFeaturesCard({ features, propertyFeatures }) {
  const allFeatures = [
    ...propertyFeatures,
    features.pool && 'Pool',
    features.garage && 'Garage',
    features.waterFront && 'Waterfront',
    features.fireplace && 'Fireplace',
    features.ac && 'Air Conditioning',
  ].filter(Boolean); // remove false values

  const uniqueFeatures = [...new Set(allFeatures.map(feature => feature.toLowerCase()))];

  return (
    <div className={styles.PropertyDetailsCard}>
      <h2>Property Features</h2>
      <ul>
        {uniqueFeatures.map((feature, index) => (
          <li key={index}>{feature}</li>
        ))}
      </ul>
    </div>
  );
}

function PropertyDescriptionCard({ remarks }) {
  return (
    <div className={styles.PropertyDetailsCard}>
      <h2>Property Description</h2>
      <p className={styles.PropertyDescription}>{remarks}</p>
    </div>
  );
}


export default function ListingDetail({bannerHeight}) {
  const router = useRouter();
  const { listingId } = router.query;

  const [data, setData] = useState(null);
  const [images, setImages] = useState([]);
  const [error, setError] = useState(null);

  const [modalVisible, setModalVisible] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const modalRef = useRef();

  // console.log('listingId:', listingId);
  const openModal = (imageUrl) => {
    setSelectedImage(imageUrl);
    setModalVisible(true);
  };
  
  const closeModal = () => {
    setModalVisible(false);
  };

  useEffect(() => {
    const fetchData = async () => {
      const apiURL = process.env.NEXT_PUBLIC_SEARCH_API_URL;
      try {
        const response = await fetch(`${apiURL}/listing/${listingId}`);
        const data = await response.json();
        setData(data);

        const imageResponse = await fetch(`${apiURL}/images/${listingId}`);
        const imagesData = await imageResponse.json();
        // setImages(imagesData.slice(0, 5));  // get 1st 5 for now
        setImages(imagesData);  // get all image urls
        // console.log('imagesData:', imagesData.slice(0, 5));

      } catch (error) {
        setError(error);
      }
    };

    if (listingId) {
      fetchData();
    }
  }, [listingId]);

  if (error) {
    // return <div>Failed to load listing</div>
    return <div aria-live="assertive">Failed to load listings. {error}</div>
  }
  if (!data) return <div>Loading...</div>

  // console.log('data:', data);

  const isDetailsAvailable = data.streetName || data.city || data.provState || data.postalCode || data.propertyType || data.transactionType || data.price || data.leasePrice || data.beds || data.baths || data.sizeInterior || data.sizeInteriorUOM || data.lotSize || data.lotUOM || data.propertyFeatures || data.remarks;

  return (
    <div className={styles.pageWrapper} style={{ marginTop: bannerHeight }}>
      {/* <h1>Listing Detail for {listingId}</h1> */}
      <div className={styles.imageWrapper}>
        {images.map((image, index) => (
          <img 
            key={index} 
            src={`${process.env.NEXT_PUBLIC_SEARCH_API_URL}/images/${image}`} 
            alt={`Listing ${listingId} Image ${index + 1}`} 
            onClick={() => openModal(`${process.env.NEXT_PUBLIC_SEARCH_API_URL}/images/${image}`)}
          />
        ))}
      </div>
      {modalVisible && (
        <div className={styles.modal} onClick={closeModal}>
          <div className={styles.modalWrapper} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalContent}>
              <img src={selectedImage} alt="Selected" className={styles.modalImage} />
              <span className={styles.close} onClick={closeModal}>&times;</span>
            </div>
          </div>
        </div>
      )}
      
      {/* {data && Object.keys(data).map(key => (
      <p key={key}>{key}: {String(data[key])}</p>
      ))} */}
      
      {isDetailsAvailable ? (
        <>
      <PropertyIdentificationCard
        jumpId={data.jumpId}
        streetName={data.streetName} 
        city={data.city}
        provState={data.provState}
        postalCode={data.postalCode}
      />
      <PropertyDetailsCard
        propertyType={data.propertyType}
        transactionType={data.transactionType}
        price={data.price}
        leasePrice={data.leasePrice}
        beds={data.beds}
        baths={data.baths}
        sizeInterior={data.sizeInterior}
        sizeInteriorUOM={data.sizeInteriorUOM}
        lotSize={data.lotSize}
        lotUOM={data.lotUOM}
      />
      {/* <PropertyFeaturesCard
        features={{
          parking: data.propertyFeatures ? data.propertyFeatures.includes('parking') : false,
          pool: data.pool,
          garage: data.garage,
          waterFront: data.waterFront,
          fireplace: data.fireplace,
          ac: data.ac
        }}
      /> */}
      <PropertyFeaturesCard 
        propertyFeatures={data.propertyFeatures || []} 
        features={{
          pool: data.pool,
          garage: data.garage,
          waterFront: data.waterFront,
          fireplace: data.fireplace,
          ac: data.ac
        }}
      />
      <PropertyDescriptionCard remarks={data.remarks} />
      </>
      ) : (
        <div className={styles.noDetails}>
          <h2>Details Not Available</h2>
          <p>Sorry, the details for this listing are not available at this time.</p>
        </div>
      ) }
    </div>
  );
}