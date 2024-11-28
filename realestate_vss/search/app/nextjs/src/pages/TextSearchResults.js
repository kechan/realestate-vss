import React from 'react';
import Link from 'next/link';
import { useTable } from 'react-table';
import styles from '../styles/TextSearchResults.module.css';

export default function TextSearchResults({ searchResults }) {

  const columns = React.useMemo(
    () => [
      {
        Header: 'Listing',
        // accessor: 'listing_id', 
        accessor: (row) => row['listing_id'] || row['listingId'], // Check for both 'listing_id' and 'listingId'
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
        Cell: ({ row, value }) => (
          <div className={styles['listing-remarks']}>
            <HighlightedRemarks 
              remarks={value}
              chunkPositions={row.original.remark_chunk_pos || []}
            />
          </div>
        ),
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
