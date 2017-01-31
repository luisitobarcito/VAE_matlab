function [data] = loadPickle(filename)
    if ~exist(filename,'file')
        error('%s is not a file',filename);
    end
    outname = [tempname() '.mat'];
    pyscript = ['import cPickle as pickle; import sys; import scipy.io; import gzip; f = gzip.open("' filename '","rb"); dat=pickle.load(f); f.close();scipy.io.savemat("' outname '",dat)'];
%     system(['LD_LIBRARY_PATH=/opt/intel/composer_xe_2013/mkl/lib/intel64:/opt/intel/composer_xe_2013/lib/intel64;python -c ''' pyscript '''']);
    system(['LD_LIBRARY_PATH=/usr/lib; python -c''' pyscript '''']);
    data = load(outname);
end