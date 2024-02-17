import React, { useState, useEffect } from 'react';
import { Button, Input, TextField, Select, MenuItem, FormControl, InputLabel, Grid, Container, Snackbar, SnackbarContent, IconButton} from '@material-ui/core';
import CloseIcon from '@material-ui/icons/Close';
import styles from '../styles/CriteriaSearchForm.module.css';

function CriteriaSearchForm({setSearchCriteria, onFormChange, searchMode}) {
  const [province, setProvince] = useState('');
  const [city, setCity] = useState('');
  const [bedsInt, setBedsInt] = useState(null);
  const [bathsInt, setBathsInt] = useState(null);
  const [minPrice, setMinPrice] = useState(null);
  const [maxPrice, setMaxPrice] = useState(null);

  const [snackbarOpen, setSnackbarOpen] = useState(false);

  useEffect(() => {
    // Ensure this code runs only in the client-side environment
    const savedFormData = JSON.parse(localStorage.getItem('criteriaSearchFormData') || '{}');

    setProvince(savedFormData.provState);
    setCity(savedFormData.city || '');
    setBedsInt(savedFormData.bedsInt || null);
    setBathsInt(savedFormData.bathsInt || null);
    setMinPrice(savedFormData.price ? savedFormData.price[0] : null);
    setMaxPrice(savedFormData.price ? savedFormData.price[1] : null);

  }, []);

  // Update the searchCriteria in the parent component whenever the form fields change
  useEffect(() => {

    const payload = {
      provState: province,
      city: city,
      bedsInt: bedsInt,
      bathsInt: bathsInt,
      price: [minPrice, maxPrice],
    };
    // Check that not all fields are empty
    const isPayloadNotEmpty = payload.provState || payload.city || payload.bedsInt !== null || payload.bathsInt !== null || payload.price.some(price => price !== null);

    if (isPayloadNotEmpty) {
      // Save to localStorage
      localStorage.setItem('criteriaSearchFormData', JSON.stringify(payload));
    } else {
      console.log('Empty payload not saved.');
    }

    setSearchCriteria(payload);
    onFormChange(payload);    // Notify the parent component about the change
  }, [province, city, bedsInt, bathsInt, minPrice, maxPrice]);
  
  // Update state only if value is a non-negative integer
  const handleIntChange = (value, setter) => {
  
    // If value is not just '0', remove leading zeros
    if (value !== '0') {
      value = value.replace(/^0+/, '');
    }
    
    if (value !== '') {
      const intValue = Math.floor(Number(value));
      if (intValue >= 0 && !isNaN(intValue)) {
        console.log('new int value:', intValue);
        setter(intValue);
      }
      else {
        console.log('new int value:', null);
        setter(null);
      }
    } else {
      setter(null);
      console.log('new int value:', null);
    }
  };

  // Update state only if value is a non-negative float
  const handleFloatChange = (value, setter) => {
    const floatValue = parseFloat(value);
    if (floatValue >= 0 && !isNaN(floatValue)) {
      setter(floatValue);
    }
  };


  const handleMinPriceChange = (value) => {
    // console.log('min price:', value)
    if (value === '') {
      setMinPrice(null);
      // console.log('new min price:', null)
    }
    else {
      const floatValue = parseFloat(value);
      // console.log('after parseFloat:', floatValue)
      if (floatValue >= 0 && !isNaN(floatValue)) {
        setMinPrice(floatValue);
        // console.log('new min price:', floatValue)
        if (floatValue > maxPrice) {
          setMaxPrice(floatValue);    // max price must be equal or greater than min price
        }
      }
    }
  };


  const handleMaxPriceChange = (event) => {
    const value = event.target.value;
    // Allow the user to freely enter text, only sanitize when focus is lost (onBlur)
    setMaxPrice(value);
  };

  const validateMaxPrice = () => {
    // Ensure maxPrice is a string before attempting to use string methods
    let maxPriceStr = String(maxPrice);
    let sanitizedValue = maxPriceStr.replace(/^0+/, '');
    sanitizedValue = sanitizedValue ? parseFloat(sanitizedValue) : '';
  
    if (!isNaN(sanitizedValue) && sanitizedValue >= 0) {
      if (sanitizedValue < minPrice) {
        setSnackbarOpen(true);
        sanitizedValue = minPrice; // Ensure maxPrice is not less than minPrice
      }
      // Convert sanitizedValue back to a string if you want to store it as a string in your state
      setMaxPrice(String(sanitizedValue));
    } else {
      setMaxPrice(''); // Set back to empty string if not a valid number
    }
  
    onFormChange(); // Notify the parent component about the change
  };
  

  const handleSnackbarClose = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }

    setSnackbarOpen(false);
  };

  return (
    <Container maxWidth="md" className={styles.formContainer}>
      {/* <form onSubmit={handleSubmit}>     */}
      <form>    
        <Grid container spacing={3}>
          <Grid item xs={12}>
            {/* <FormControl variant="outlined" fullWidth className={`${styles.formControl} ${styles.noIndicator} ${styles.noNotch} ${styles.noBorder}`}> */}
            <FormControl variant="outlined" fullWidth className={`${styles.formControl} ${styles.noIndicator} ${styles.noNotch} ${styles.noBorder}`}>
              <InputLabel id="province-label">Province</InputLabel>
              <Select labelId="province-label" value={province} onChange={e => setProvince(e.target.value)} className={styles.selectEmpty}>
                <MenuItem value="ON">Ontario</MenuItem>
                <MenuItem value="BC">British Columbia</MenuItem>
                <MenuItem value="AB">Alberta</MenuItem>
                <MenuItem value="SK">Saskatchewan</MenuItem>
                <MenuItem value="MB">Manitoba</MenuItem>
                <MenuItem value="QC">Quebec</MenuItem>
                <MenuItem value="NB">New Brunswick</MenuItem>
                <MenuItem value="NS">Nova Scotia</MenuItem>
                <MenuItem value="PE">Prince Edward Island</MenuItem>
                <MenuItem value="NL">Newfoundland and Labrador</MenuItem>
                <MenuItem value="YT">Yukon</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          {searchMode !== 'VSS_ONLY' && (
            <>
              <Grid item xs={12}>
                <FormControl fullWidth>
                    <InputLabel htmlFor="city">City</InputLabel>
                    <Input id="city" type="text" value={city} onChange={e => setCity(e.target.value)} />
                  </FormControl>   
              </Grid>
              <Grid item xs={6}>
                  <FormControl fullWidth>
                    <InputLabel htmlFor="bedsInt">Beds</InputLabel>
                    <Input id="bedsInt" type="text" 
                      value={bedsInt === null ? '' : bedsInt}
                      onChange={e => handleIntChange(e.target.value, setBedsInt)} 
                      inputProps= {{pattern: "[0-9]*"}}
                    />
                  </FormControl>
                </Grid>
                <Grid item xs={6}>
                  <FormControl fullWidth>
                    <InputLabel htmlFor="bathsInt">Baths</InputLabel>
                    <Input id="bathsInt" type="text" 
                    value={bathsInt === null ? '' : bathsInt} 
                    onChange={e => handleIntChange(e.target.value, setBathsInt)} 
                    inputProps= {{pattern: "[0-9]*"}}
                    />
                  </FormControl>
                </Grid>
              <Grid item xs={6}>          
                <FormControl fullWidth>
                    <InputLabel htmlFor="minPrice">Min Price</InputLabel>
                    <Input id="minPrice" type="text" 
                    value={minPrice === null ? '' : minPrice} 
                    onChange={e => handleMinPriceChange(e.target.value)} 
                    inputProps={{pattern: "[0-9]*"}}
                    />
                </FormControl>
              </Grid>
              <Grid item xs={6}>          
                  <FormControl fullWidth>
                    <InputLabel htmlFor="maxPrice" shrink={maxPrice !== null}>Max Price</InputLabel>
                    <Input id="maxPrice" type="text" 
                      value={maxPrice}
                      onChange={handleMaxPriceChange}
                      onBlur={validateMaxPrice}
                      inputProps={{pattern: "[0-9]*"}}
                    />
                </FormControl>
              </Grid>
            </>
          )}
        </Grid>        
      </form>

      <Snackbar
        open={snackbarOpen}
        autoHideDuration={3000}
        onClose={handleSnackbarClose}
        message="Max price cannot be less than min price"        
        action={
          <IconButton size="small" aria-label="close" color="inherit" onClick={handleSnackbarClose}>
            <CloseIcon fontSize="small" />
          </IconButton>
        }
      >
        <SnackbarContent style={{
          backgroundColor: '#D2232A',
        }}
        message={<span id="client-snackbar">Max price cannot be less than min price</span>}
        />
      </Snackbar>
    </Container>
  );
}

export default CriteriaSearchForm;