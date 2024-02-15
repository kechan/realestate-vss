import React, {useEffect, useState} from 'react';
import Link from 'next/link';
import { useTable } from 'react-table';
import styles from '../styles/TextSearchResults.module.css';

export default function TextSearchResults({ searchResults }) {
  // Initialize localSearchResults as an empty array
  const [localSearchResults, setLocalSearchResults] = useState([]);

  // Load search results from Local Storage after initial render
  useEffect(() => {
    const savedResults = localStorage.getItem('searchResults');
    if (savedResults) {
      // If there are search results in Local Storage, parse them and set them
      setLocalSearchResults(JSON.parse(savedResults));
    } else {
      // If there are no search results in Local Storage, use the prop
      setLocalSearchResults(searchResults);
    }
  }, []);

  // Save search results to Local Storage whenever they change
  useEffect(() => {
    localStorage.setItem('searchResults', JSON.stringify(localSearchResults));
  }, [localSearchResults]);

  const columns = React.useMemo(
    () => [
      {
        Header: 'Listing',
        accessor: 'listing_id', // Ensure the accessor matches the corresponding key in your data
        Cell: ({ value }) => (
          <Link href={`/listing/${value}`} className={styles['listing-link']}>
            {value}
          </Link>
        ),
      },
      {
        Header: 'Score',
        accessor: 'score',
        Cell: ({ value }) => value ? parseFloat(value).toFixed(2) : 'N/A',
      },
      {
        Header: 'City',
        accessor: 'city',
      },
      {
        Header: 'Beds',
        accessor: 'beds',
      },
      {
        Header: 'Baths',
        accessor: 'baths',
      },
      {
        Header: 'StreetName',
        accessor: 'streetName',
      },
      {
        Header: 'Remarks',
        accessor: 'remarks',
        Cell: ({ value }) => (
          <div className={styles['listing-remarks']}>{value}</div>
        ),
        // headerClassName: styles['header-remarks'],
      }
      
    ],
    []
  );

  const tableInstance = useTable({ columns, data: searchResults });

  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    rows,
    prepareRow,
  } = tableInstance;

  return (
    <div className={styles.ReactTable} {...getTableProps()}>
      <div className={styles.thead}>
        {headerGroups.map(headerGroup => (
          <div className={styles.tr} {...headerGroup.getHeaderGroupProps()}>
            {headerGroup.headers.map(column => (
              <div
                className={column.id === 'remarks' ? styles['header-remarks'] : styles.th}
                {...column.getHeaderProps()}
              >
                {column.render('Header')}
              </div>
            ))}
          </div>
        ))}
      </div>
      <div className={styles.tbody} {...getTableBodyProps()}>
        {rows.map(row => {
          prepareRow(row);
          return (
            <div className={styles.tr} {...row.getRowProps()}>
              {row.cells.map(cell => (
                <div
                  className={cell.column.id === 'remarks' ? styles['listing-remarks'] : styles.td}
                  {...cell.getCellProps()}
                >
                  {cell.render('Cell')}
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}
