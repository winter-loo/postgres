/*-------------------------------------------------------------------------
 *
 * toasting.h
 *	  This file provides some definitions to support creation of toast tables
 *
 *
 * Portions Copyright (c) 1996-2023, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 * src/include/catalog/toasting.h
 *
 *-------------------------------------------------------------------------
 */
#ifndef TOASTING_H
#define TOASTING_H

#include "storage/lock.h"

#define ITIME_TABLE_NAME_PREFIX "pg_itime_"
#define ITIME_INDEX_ITIME_PREFIX "pg_itime_index_itime_"
#define ITIME_INDEX_CTID_PREFIX "pg_itime_index_ctid_"
#define Aname_pg_itime_xxx_itime "itime_"
#define Aname_pg_itime_xxx_ctid "ctid_"
#define Anum_pg_itime_xxx_itime 	1
#define Anum_pg_itime_xxx_ctid 	2


/*
 * toasting.c prototypes
 */
extern void NewRelationCreateToastTable(Oid relOid, Datum reloptions);
extern void NewHeapCreateToastTable(Oid relOid, Datum reloptions,
									LOCKMODE lockmode, Oid OIDOldToast);
extern void AlterTableCreateToastTable(Oid relOid, Datum reloptions,
									   LOCKMODE lockmode);
extern void BootstrapToastTable(char *relName,
								Oid toastOid, Oid toastIndexOid);

extern void NewRelationCreateItimeTable(Oid relOid);

#endif							/* TOASTING_H */
