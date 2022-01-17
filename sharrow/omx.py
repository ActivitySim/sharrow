import logging
import os

logger = logging.getLogger("sharrow.omx")


def split_omx(source_file, dest_directory, global_lookups=False, n_chunks=None):
    """
    Split an OMX file into separate files for each element.

    Parameters
    ----------
    source_file : str
    dest_directory : str
    global_lookups : bool

    """
    from larch import OMX

    source = OMX(source_file, mode="r")
    os.makedirs(dest_directory, exist_ok=True)

    if n_chunks is not None:
        general_name = os.path.splitext(source_file)[0]
        chunkfiles = [f"{general_name}-chunk{n}.omx" for n in range(n_chunks)]
    else:
        chunkfiles = [f"{k}.omx" for k in list(source.data._v_children)]

    n = 0
    for k in list(source.data._v_children):
        newfile = os.path.join(
            dest_directory,
            chunkfiles[n],
        )
        logger.info(f"writing {k} to {newfile}")
        b = OMX(newfile, mode="a")
        b.add_matrix(k, source.data[k], overwrite=True)
        if global_lookups:
            # todo only write once per file
            for j in list(source.lookup._v_children):
                b.add_lookup(j, source.lookup[j], overwrite=True)
        b.close()
        n += 1
        if n >= len(chunkfiles):
            n = 0
    if not global_lookups:
        for k in list(source.lookup._v_children):
            newfile = os.path.join(
                dest_directory,
                f"_{k}.omx",
            )
            logger.info(f"writing {newfile}")
            b = OMX(newfile, mode="w")
            b.shape = source.shape
            b.add_lookup(k, source.lookup[k])
            b.close()
