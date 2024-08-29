 def test_successful_parallel_download(self, mocker):
        from clfGraph.data.dl_script import (download_chunk_parallel,
                                             download_session)
        from clfGraph.src.custom_logging import logger

        # Mock the download_chunk method to simulate successful download
        mocker.patch.object(download_session, 'download_chunk', return_value=None)
    
        # Mock the logger to avoid actual logging during tests
        mocker.patch.object(logger, 'info')
        mocker.patch.object(logger, 'error')
    
        # Call the function to test
        download_chunk_parallel(dl_session=download_session)
    
        # Assert that the download_chunk method was called the expected number of times
        assert download_session.download_chunk.call_count == len(download_session.file_list) // download_session.chunk_size + 1
    
        # Assert that no errors were logged
        logger.error.assert_not_called()


