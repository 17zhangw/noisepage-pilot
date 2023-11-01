//
// our actual bindings modules -- these are generated by build.rs
//

// feature gate each pg version module
#[cfg(all(feature = "pg11", not(docsrs)))]
pub(crate) mod pg11 {
    include!(concat!(env!("OUT_DIR"), "/pg11.rs"));
}
#[cfg(all(feature = "pg11", docsrs))]
pub(crate) mod pg11;

#[cfg(all(feature = "pg12", not(docsrs)))]
pub(crate) mod pg12 {
    include!(concat!(env!("OUT_DIR"), "/pg12.rs"));
}
#[cfg(all(feature = "pg12", docsrs))]
pub(crate) mod pg12;

#[cfg(all(feature = "pg13", not(docsrs)))]
pub(crate) mod pg13 {
    include!(concat!(env!("OUT_DIR"), "/pg13.rs"));
}
#[cfg(all(feature = "pg13", docsrs))]
pub(crate) mod pg13;

#[cfg(all(feature = "pg14", not(docsrs)))]
pub(crate) mod pg14 {
    include!(concat!(env!("OUT_DIR"), "/pg14.rs"));
}
#[cfg(all(feature = "pg14", docsrs))]
pub(crate) mod pg14;

#[cfg(all(feature = "pg15", not(docsrs)))]
pub(crate) mod pg15 {
    include!(concat!(env!("OUT_DIR"), "/pg15.rs"));
}
#[cfg(all(feature = "pg15", docsrs))]
pub(crate) mod pg15;

#[cfg(feature = "cpg15")]
pub(crate) mod cpg15;

#[cfg(all(feature = "pg16", not(docsrs)))]
pub(crate) mod pg16 {
    include!(concat!(env!("OUT_DIR"), "/pg16.rs"));
}
#[cfg(all(feature = "pg16", docsrs))]
pub(crate) mod pg16;

// export each module publicly
#[cfg(feature = "pg11")]
pub use pg11::*;
#[cfg(feature = "pg12")]
pub use pg12::*;
#[cfg(feature = "pg13")]
pub use pg13::*;
#[cfg(feature = "pg14")]
pub use pg14::*;
#[cfg(feature = "pg15")]
pub use pg15::*;
#[cfg(feature = "cpg15")]
pub use cpg15::*;
#[cfg(feature = "pg16")]
pub use pg16::*;

// feature gate each pg-specific oid module
#[cfg(all(feature = "pg11", not(docsrs)))]
mod pg11_oids {
    include!(concat!(env!("OUT_DIR"), "/pg11_oids.rs"));
}
#[cfg(all(feature = "pg11", docsrs))]
mod pg11_oids;

#[cfg(all(feature = "pg12", not(docsrs)))]
mod pg12_oids {
    include!(concat!(env!("OUT_DIR"), "/pg12_oids.rs"));
}
#[cfg(all(feature = "pg12", docsrs))]
mod pg12_oids;

#[cfg(all(feature = "pg13", not(docsrs)))]
mod pg13_oids {
    include!(concat!(env!("OUT_DIR"), "/pg13_oids.rs"));
}
#[cfg(all(feature = "pg13", docsrs))]
mod pg13_oids;

#[cfg(all(feature = "pg14", not(docsrs)))]
mod pg14_oids {
    include!(concat!(env!("OUT_DIR"), "/pg14_oids.rs"));
}
#[cfg(all(feature = "pg14", docsrs))]
mod pg14_oids;

#[cfg(all(feature = "pg15", not(docsrs)))]
mod pg15_oids {
    include!(concat!(env!("OUT_DIR"), "/pg15_oids.rs"));
}
#[cfg(all(feature = "pg15", docsrs))]
mod pg15_oids;

#[cfg(feature = "cpg15")]
mod cpg15_oids;

#[cfg(all(feature = "pg16", not(docsrs)))]
mod pg16_oids {
    include!(concat!(env!("OUT_DIR"), "/pg16_oids.rs"));
}
#[cfg(all(feature = "pg16", docsrs))]
mod pg16_oids;

// export that module publicly
#[cfg(feature = "pg11")]
pub use pg11_oids::*;
#[cfg(feature = "pg12")]
pub use pg12_oids::*;
#[cfg(feature = "pg13")]
pub use pg13_oids::*;
#[cfg(feature = "pg14")]
pub use pg14_oids::*;
#[cfg(feature = "pg15")]
pub use pg15_oids::*;
#[cfg(feature = "cpg15")]
pub use cpg15_oids::*;
#[cfg(feature = "pg16")]
pub use pg16_oids::*;

mod internal {
    //
    // for specific versions
    //
    #[cfg(feature = "pg11")]
    pub(crate) mod pg11 {
        pub use crate::pg11::tupleDesc as TupleDescData;
        pub type QueryCompletion = std::os::raw::c_char;

        /// # Safety
        ///
        /// This function wraps Postgres' internal `IndexBuildHeapScan` method, and therefore, is
        /// inherently unsafe
        pub unsafe fn IndexBuildHeapScan<T>(
            heap_relation: crate::Relation,
            index_relation: crate::Relation,
            index_info: *mut crate::pg11::IndexInfo,
            build_callback: crate::IndexBuildCallback,
            build_callback_state: *mut T,
        ) {
            crate::pg11::IndexBuildHeapScan(
                heap_relation,
                index_relation,
                index_info,
                true,
                build_callback,
                build_callback_state as *mut std::os::raw::c_void,
                std::ptr::null_mut(),
            );
        }
    }

    #[cfg(feature = "pg12")]
    pub(crate) mod pg12 {
        pub use crate::pg12::AllocSetContextCreateInternal as AllocSetContextCreateExtended;
        pub type QueryCompletion = std::os::raw::c_char;

        pub const QTW_EXAMINE_RTES: u32 = crate::pg12::QTW_EXAMINE_RTES_BEFORE;

        /// # Safety
        ///
        /// This function wraps Postgres' internal `IndexBuildHeapScan` method, and therefore, is
        /// inherently unsafe
        pub unsafe fn IndexBuildHeapScan<T>(
            heap_relation: crate::Relation,
            index_relation: crate::Relation,
            index_info: *mut crate::pg12::IndexInfo,
            build_callback: crate::IndexBuildCallback,
            build_callback_state: *mut T,
        ) {
            let heap_relation_ref = heap_relation.as_ref().unwrap();
            let table_am = heap_relation_ref.rd_tableam.as_ref().unwrap();

            table_am.index_build_range_scan.unwrap()(
                heap_relation,
                index_relation,
                index_info,
                true,
                false,
                true,
                0,
                crate::InvalidBlockNumber,
                build_callback,
                build_callback_state as *mut std::os::raw::c_void,
                std::ptr::null_mut(),
            );
        }
    }

    #[cfg(feature = "pg13")]
    pub(crate) mod pg13 {
        pub use crate::pg13::AllocSetContextCreateInternal as AllocSetContextCreateExtended;

        pub const QTW_EXAMINE_RTES: u32 = crate::pg13::QTW_EXAMINE_RTES_BEFORE;

        /// # Safety
        ///
        /// This function wraps Postgres' internal `IndexBuildHeapScan` method, and therefore, is
        /// inherently unsafe
        pub unsafe fn IndexBuildHeapScan<T>(
            heap_relation: crate::Relation,
            index_relation: crate::Relation,
            index_info: *mut crate::IndexInfo,
            build_callback: crate::IndexBuildCallback,
            build_callback_state: *mut T,
        ) {
            let heap_relation_ref = heap_relation.as_ref().unwrap();
            let table_am = heap_relation_ref.rd_tableam.as_ref().unwrap();

            table_am.index_build_range_scan.unwrap()(
                heap_relation,
                index_relation,
                index_info,
                true,
                false,
                true,
                0,
                crate::InvalidBlockNumber,
                build_callback,
                build_callback_state as *mut std::os::raw::c_void,
                std::ptr::null_mut(),
            );
        }
    }

    #[cfg(feature = "pg14")]
    pub(crate) mod pg14 {
        pub use crate::pg14::AllocSetContextCreateInternal as AllocSetContextCreateExtended;

        pub const QTW_EXAMINE_RTES: u32 = crate::pg14::QTW_EXAMINE_RTES_BEFORE;

        /// # Safety
        ///
        /// This function wraps Postgres' internal `IndexBuildHeapScan` method, and therefore, is
        /// inherently unsafe
        pub unsafe fn IndexBuildHeapScan<T>(
            heap_relation: crate::Relation,
            index_relation: crate::Relation,
            index_info: *mut crate::IndexInfo,
            build_callback: crate::IndexBuildCallback,
            build_callback_state: *mut T,
        ) {
            let heap_relation_ref = heap_relation.as_ref().unwrap();
            let table_am = heap_relation_ref.rd_tableam.as_ref().unwrap();

            table_am.index_build_range_scan.unwrap()(
                heap_relation,
                index_relation,
                index_info,
                true,
                false,
                true,
                0,
                crate::InvalidBlockNumber,
                build_callback,
                build_callback_state as *mut std::os::raw::c_void,
                std::ptr::null_mut(),
            );
        }
    }

    #[cfg(any(feature = "pg15", feature = "cpg15"))]
    pub(crate) mod pg15 {
        #[cfg(feature = "cpg15")]
        pub use crate::cpg15::AllocSetContextCreateInternal as AllocSetContextCreateExtended;
        #[cfg(feature = "pg15")]
        pub use crate::pg15::AllocSetContextCreateInternal as AllocSetContextCreateExtended;

        #[cfg(feature = "pg15")]
        pub const QTW_EXAMINE_RTES: u32 = crate::pg15::QTW_EXAMINE_RTES_BEFORE;
        #[cfg(feature = "cpg15")]
        pub const QTW_EXAMINE_RTES: u32 = crate::cpg15::QTW_EXAMINE_RTES_BEFORE;

        /// # Safety
        ///
        /// This function wraps Postgres' internal `IndexBuildHeapScan` method, and therefore, is
        /// inherently unsafe
        pub unsafe fn IndexBuildHeapScan<T>(
            heap_relation: crate::Relation,
            index_relation: crate::Relation,
            index_info: *mut crate::IndexInfo,
            build_callback: crate::IndexBuildCallback,
            build_callback_state: *mut T,
        ) {
            let heap_relation_ref = heap_relation.as_ref().unwrap();
            let table_am = heap_relation_ref.rd_tableam.as_ref().unwrap();

            table_am.index_build_range_scan.unwrap()(
                heap_relation,
                index_relation,
                index_info,
                true,
                false,
                true,
                0,
                crate::InvalidBlockNumber,
                build_callback,
                build_callback_state as *mut std::os::raw::c_void,
                std::ptr::null_mut(),
            );
        }
    }

    #[cfg(feature = "pg16")]
    pub(crate) mod pg16 {
        pub use crate::pg16::AllocSetContextCreateInternal as AllocSetContextCreateExtended;

        pub const QTW_EXAMINE_RTES: u32 = crate::pg16::QTW_EXAMINE_RTES_BEFORE;

        /// # Safety
        ///
        /// This function wraps Postgres' internal `IndexBuildHeapScan` method, and therefore, is
        /// inherently unsafe
        pub unsafe fn IndexBuildHeapScan<T>(
            heap_relation: crate::Relation,
            index_relation: crate::Relation,
            index_info: *mut crate::IndexInfo,
            build_callback: crate::IndexBuildCallback,
            build_callback_state: *mut T,
        ) {
            let heap_relation_ref = heap_relation.as_ref().unwrap();
            let table_am = heap_relation_ref.rd_tableam.as_ref().unwrap();

            table_am.index_build_range_scan.unwrap()(
                heap_relation,
                index_relation,
                index_info,
                true,
                false,
                true,
                0,
                crate::InvalidBlockNumber,
                build_callback,
                build_callback_state as *mut std::os::raw::c_void,
                std::ptr::null_mut(),
            );
        }
    }
}

// and things that are version-specific
#[cfg(feature = "pg11")]
pub use internal::pg11::IndexBuildHeapScan;
#[cfg(feature = "pg11")]
pub use internal::pg11::*;

#[cfg(feature = "pg12")]
pub use internal::pg12::*;

#[cfg(feature = "pg13")]
pub use internal::pg13::*;

#[cfg(feature = "pg14")]
pub use internal::pg14::*;

#[cfg(feature = "pg15")]
pub use internal::pg15::*;

#[cfg(feature = "cpg15")]
pub use internal::pg15::*;

#[cfg(feature = "pg16")]
pub use internal::pg16::*;
