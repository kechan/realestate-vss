import React, { useState } from 'react';
import { Button, Input, TextField, Select, MenuItem, FormControl, InputLabel, Grid, Container, Snackbar, IconButton} from '@material-ui/core';
import CloseIcon from '@material-ui/icons/Close';
import styles from '../styles/CriteriaSearchForm.module.css';

function CriteriaSearchForm() {
  const [province, setProvince] = useState('');
  const [city, setCity] = useState('');
  const [bedsInt, setBedsInt] = useState(null);
  const [bathsInt, setBathsInt] = useState(null);
  const [minPrice, setMinPrice] = useState(null);
  const [maxPrice, setMaxPrice] = useState(null);

  const [snackbarOpen, setSnackbarOpen] = useState(false);
  
  // Update state only if value is a non-negative integer
  const handleIntChange = (value, setter) => {
    const intValue = Math.floor(Number(value));
    if (intValue >= 0 && !isNaN(intValue)) {
      setter(intValue);
      console.log(`New value:', ${intValue}`);
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
    const floatValue = parseFloat(value);
    if (floatValue >= 0 && !isNaN(floatValue)) {
      setMinPrice(floatValue);
      if (floatValue > maxPrice) {
        setMaxPrice(floatValue);    // max price must be equal or greater than min price
      }
    }
  };

  const handleMaxPriceChange = (value) => {
    const floatValue = parseFloat(value);
    if (!isNaN(floatValue)) {
      setMaxPrice(floatValue);
    }
  };

  const validateMaxPrice = () => {
    if (maxPrice < minPrice) {
      setMaxPrice(minPrice);
      setSnackbarOpen(true);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const payload = {
      provState: province,
      city: city,
      bedsInt: bedsInt,
      bathsInt: bathsInt,
      price: [minPrice, maxPrice],
    };
    // Call API with payload
  };

  const handleSnackbarClose = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }

    setSnackbarOpen(false);
  };

  return (
    <Container maxWidth="md" className={styles.formContainer}>
      <form onSubmit={handleSubmit}>    
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
          <Grid item xs={12}>
            <FormControl fullWidth>
                <InputLabel htmlFor="city">City</InputLabel>
                <Input id="city" type="number" value={city} onChange={e => setCity(e.target.value)} />
              </FormControl>   
          </Grid>
          <Grid item xs={6}>
              <FormControl fullWidth>
                <InputLabel htmlFor="bedsInt">Beds</InputLabel>
                <Input id="bedsInt" type="number" value={bedsInt} onChange={e => handleIntChange(e.target.value, setBedsInt)} 
                inputProps= {{min: 0}}
                />
              </FormControl>
            </Grid>
            <Grid item xs={6}>
              <FormControl fullWidth>
                <InputLabel htmlFor="bathsInt">Baths</InputLabel>
                <Input id="bathsInt" type="number" value={bathsInt} onChange={e => handleIntChange(e.target.value, setBathsInt)} 
                inputProps= {{min: 0}}
                />
              </FormControl>
            </Grid>
          <Grid item xs={6}>          
            <FormControl fullWidth>
                <InputLabel htmlFor="minPrice">Min Price</InputLabel>
                <Input id="minPrice" type="number" value={minPrice} onChange={e => handleMinPriceChange(e.target.value)} 
                inputProps={{ min: 0 }}
                />
            </FormControl>
          </Grid>
          <Grid item xs={6}>          
              <FormControl fullWidth>
                <InputLabel htmlFor="maxPrice" shrink={maxPrice !== null}>Max Price</InputLabel>
                <Input id="maxPrice" type="number" 
                  value={maxPrice} 
                  onChange={e => handleMaxPriceChange(e.target.value)} 
                  onBlur={validateMaxPrice}
                  inputProps={{ min: 0 }}
                />
            </FormControl>
          </Grid>
          <Grid item xs={12}>
            {/* <Button type="submit" variant="contained" color="primary" fullWidth>Search</Button> */}
          </Grid>
        </Grid>
        
      </form>

      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        message="Max price cannot be less than min price"
        action={
          <IconButton size="small" aria-label="close" color="inherit" onClick={handleSnackbarClose}>
            <CloseIcon fontSize="small" />
          </IconButton>
        }
      />
    </Container>
  );
}

export default CriteriaSearchForm;