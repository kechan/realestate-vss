import '@/styles/globals.css'
import Head from 'next/head'

import React, { useRef, useEffect, useState } from 'react';
import Banner from './Banner' 

export default function App({ Component, pageProps }) {
  const bannerRef = useRef(null);
  const [bannerHeight, setBannerHeight] = useState(0);

  useEffect(() => {
    if (bannerRef.current) {
      setBannerHeight(bannerRef.current.offsetHeight);
    }
  }, []);

  return (
    <>
      <Head>
        <link href="https://fonts.googleapis.com/css2?family=Exo+2:wght@400;700&display=swap" rel="stylesheet" />
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet" />
     
        {/* You can include any other global tags here */}
      </Head>
      <Banner ref={bannerRef}/>
      <Component {...pageProps} bannerHeight={bannerHeight}/>
    </>
  )
}
