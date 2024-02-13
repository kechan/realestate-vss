import { useRouter } from 'next/router';
import { useEffect, useState } from 'react';
import styles from '../../styles/[listingId].module.css';

export default function ListingDetail({bannerHeight}) {
  const router = useRouter();
  const { listingId } = router.query;

  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  // console.log('listingId:', listingId);

  useEffect(() => {
    const fetchData = async () => {
      const apiURL = process.env.NEXT_PUBLIC_SEARCH_API_URL;
      try {
        const response = await fetch(`${apiURL}/listing/${listingId}`);
        const data = await response.json();
        setData(data);
      } catch (error) {
        setError(error);
      }
    };

    if (listingId) {
      fetchData();
    }
  }, [listingId]);

  if (error) {
    // console.log('Error:', error)
    return <div>Failed to load listing</div>
  }
  if (!data) return <div>Loading...</div>

  // console.log('data:', data);

  return (
    <div className={styles.pageWrapper} style={{ marginTop: bannerHeight }}>
      <h1>Listing Detail for {listingId}</h1>
      {/* {data && (
        <>
          <p>City: {data.city}</p>
          <p>Street Name: {data.streetName}</p>
          <p>Beds: {data.beds}</p>
          <p>Baths: {data.baths}</p>
          <p>Price: {data.price}</p>
        </>
      )} */}
      {data && Object.keys(data).map(key => (
      <p key={key}>{key}: {String(data[key])}</p>
      ))}
    </div>
  );
}