/*-------------------------------------------------------------------------
 *
 * nodeSeqscan.h
 *
 *
 *
 * Portions Copyright (c) 1996-2022, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 * src/include/executor/nodeSeqscan.h
 *
 *-------------------------------------------------------------------------
 */
#ifndef NODESEQSCAN_H
#define NODESEQSCAN_H

#include "access/parallel.h"
#include "nodes/execnodes.h"

typedef TupleTableSlot *(*SeqNext_hook_type) (SeqScanState *node);
extern PGDLLIMPORT SeqNext_hook_type SeqNext_hook;

extern SeqScanState *ExecInitSeqScan(SeqScan *node, EState *estate, int eflags);
extern void ExecEndSeqScan(SeqScanState *node);
extern void ExecReScanSeqScan(SeqScanState *node);
extern TupleTableSlot *SeqNext(SeqScanState *node);
extern TupleTableSlot *standard_SeqNext(SeqScanState *node);

/* parallel scan support */
extern void ExecSeqScanEstimate(SeqScanState *node, ParallelContext *pcxt);
extern void ExecSeqScanInitializeDSM(SeqScanState *node, ParallelContext *pcxt);
extern void ExecSeqScanReInitializeDSM(SeqScanState *node, ParallelContext *pcxt);
extern void ExecSeqScanInitializeWorker(SeqScanState *node,
										ParallelWorkerContext *pwcxt);

#endif							/* NODESEQSCAN_H */
