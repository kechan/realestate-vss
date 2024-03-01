import { useRouter } from 'next/router';
import { useEffect, useState } from 'react';
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

function PropertyFeaturesCard({ features }) {
  return (
    <div className={styles.PropertyDetailsCard}>
      <h2>Property Features</h2>
      <ul>
        {features.parking && <li>Parking</li>}
        {features.pool && <li>Pool</li>}
        {features.garage && <li>Garage</li>}
        {features.waterFront && <li>Waterfront</li>}
        {features.fireplace && <li>Fireplace</li>}
        {features.ac && <li>Air Conditioning</li>}
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

  // console.log('listingId:', listingId);

  useEffect(() => {
    const fetchData = async () => {
      const apiURL = process.env.NEXT_PUBLIC_SEARCH_API_URL;
      try {
        const response = await fetch(`${apiURL}/listing/${listingId}`);
        const data = await response.json();
        setData(data);

        const imageResponse = await fetch(`${apiURL}/images/${listingId}`);
        const imagesData = await imageResponse.json();
        setImages(imagesData.slice(0, 5));  // get 1st 5 for now
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

  return (
    <div className={styles.pageWrapper} style={{ marginTop: bannerHeight }}>
      {/* <h1>Listing Detail for {listingId}</h1> */}
      <div className={styles.imageWrapper}>
        {images.map((image, index) => (
          <img key={index} src={`${process.env.NEXT_PUBLIC_SEARCH_API_URL}/images/${image}`} alt={`Listing ${listingId} Image ${index + 1}`} />
        ))}
      </div>
      {/* {data && (
        <>
          <p>City: {data.city}</p>
          <p>Street Name: {data.streetName}</p>
          <p>Beds: {data.beds}</p>
          <p>Baths: {data.baths}</p>
          <p>Price: {data.price}</p>
        </>
      )} */}
      {/* {data && Object.keys(data).map(key => (
      <p key={key}>{key}: {String(data[key])}</p>
      ))} */}
      
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
      <PropertyFeaturesCard
        features={{
          parking: data.propertyFeatures.includes('parking'),
          pool: data.pool,
          garage: data.garage,
          waterFront: data.waterFront,
          fireplace: data.fireplace,
          ac: data.ac
        }}
      />
      <PropertyDescriptionCard remarks={data.remarks} />
    </div>
  );
}