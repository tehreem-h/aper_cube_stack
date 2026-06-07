# Some lines adopted from Tom Oosterloo's findBeam.py

from argparse import ArgumentParser, RawTextHelpFormatter
import sys
import os

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, join, unique
import astropy.units as u
from astropy.wcs import WCS
import numpy as np

from regrid_aperpb import model_lookup, get_cb_model_freq


###################################################################

def parse_args():
    parser = ArgumentParser(description="Find the best psf values for each individual detection.",
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('-f', '--catalog_file', default=None,
                        help='Specify the source catalog file for which the beam info should be determined.')

    parser.add_argument('-c', '--cube', default='2', type=int,
                        help='Specify the cubes on which to do source finding (default: %(default)s).')

    parser.add_argument('-p', '--pb_root_dir', default=None,
                        help='Specify the root directory where the primary beam models are located (default: %(default)s).')

    # Parse the arguments above
    arguments = parser.parse_args()
    return arguments


def avg_cube_freq(cube):
    # Frequencies listed in order of ascending cube number:
    # Is the avg_cube_freq wrong?  NAXIS3 should be /2? Copied based on regrid_aperpb.py  
    #   Difference in scaling is only 1.5%, soooo no worries?
    f1 = (1293922618.880713 + 3.66210937500e4 * 1175) * u.Hz
    f2 = (1333251485.36422 + 3.66210937500e4 * 1212) * u.Hz
    f3 = (1373904086.115967 + 3.66210937500e4 * 1212) * u.Hz
    f4 = (1414373157.059737 + 3.66210937500e4 * 1218) * u.Hz
    frequencies = [f1, f2, f3, f4]
    return frequencies[cube]


# def pb_weight(ra, dec, pointing, cube, pb_root_dir='/mnt/scratch/apertif/cbeams/'):
def pb_weight(ra, dec, pointing, cube, pb_root_dir=''):
    pb_weights = []
    for ptg in pointing:
        # Repeats some code from regrid_aperbp.py
        pb_model = model_lookup(ptg['beam'], root_dir=pb_root_dir)
        hdulist_pb = fits.open(pb_model)
        hdulist_pb[0].header['CRVAL1'] = ptg['RA']
        hdulist_pb[0].header['CRVAL2'] = ptg['Dec']

        #avg_cube_freq = (temp_header['CRVAL3'] + temp_header['CDELT3'] * temp_header['NAXIS3']) * u.Hz
        cube_freq = avg_cube_freq(cube)

        hdulist_pb[0].header['CDELT1'] = (
                    hdulist_pb[0].header['CDELT1'] * get_cb_model_freq().to(u.Hz) / cube_freq).value
        hdulist_pb[0].header['CDELT2'] = (
                    hdulist_pb[0].header['CDELT2'] * get_cb_model_freq().to(u.Hz) / cube_freq).value

        x, y = WCS(hdulist_pb[0].header).world_to_pixel(SkyCoord(ra, dec, unit='deg'))
        pb_weights.append(hdulist_pb[0].data[int(y), int(x)])

    return np.array(pb_weights)


def get_psf_per_chan(source, im_cube):

    psf_hdu = fits.open(im_cube)
    if len(psf_hdu[1].data) == len(psf_hdu[0].data[:,0,0]):
        psf_chans = psf_hdu[1].data[source['z_min']:source['z_max']+1]
    else:
        psf_chans = None
        print("Need better chan code for {}".format(im_cube))
    psf_hdu.close()

    return psf_chans


def get_psf_per_field(field, cube):

    filename = 'mos_{0}/{0}_HIcube{1}_clean_image.fits'.format(field, cube)
    hdr = fits.getheader(filename)
    psf = {'bmaj_field':hdr['BMAJ'], 'bmin_field':hdr['BMIN'], 'bpa_field':hdr['BPA']}

    return psf


def main(source, field, cube, ptgs, pb_root_dir=''):
    
    fac = np.cos(source['dec'] * np.pi / 180.0)
    dist = np.sqrt((source['ra'] - ptgs['RA'])**2 * fac * fac + (source['dec'] - ptgs['Dec'])**2)

    # The smallest subset of pointings.dat within maxRad of the target galaxy and the correct field:
    maxRad = 0.5
    ptgs = ptgs[dist < maxRad]
    dist = dist[dist < maxRad]
    
    forgotten_beams = []
    if len(ptgs) > 0:
        indexes = np.argsort(dist)
        ptgs = ptgs[indexes]
        dist = dist[indexes]

        # Get the pb values at the location of the source for all beams where the contribution is greater than 25% closest beam.
        pb_values = pb_weight(source['ra'], source['dec'], ptgs, cube, pb_root_dir=pb_root_dir)
        ptgs2 = ptgs[pb_values > 0.25*pb_values[0]]
        dist2 = dist[pb_values > 0.25*pb_values[0]]
        pb_values = pb_values[pb_values > 0.25*pb_values[0]]

        psf_cubes = [field + '/HI_B0{:02}_cube{}_spline_clean_image.fits'.format(b, cube) for b in ptgs2['beam']]

        for i in range(len(psf_cubes)-1, -1, -1):
            if not os.path.isfile(psf_cubes[i]):
                forgotten_beams.append([source['name'][0], psf_cubes[i], i])
                psf_cubes.pop(i)
                ptgs2.remove_row(i)
                pb_values = np.delete(pb_values, i)

        psf_per_chan = [get_psf_per_chan(source, p) for p in psf_cubes]

        psf_nearest = {'beam':ptgs2[0]['beam'],'bmaj_med':np.median(psf_per_chan[0]['BMAJ']), 'bmin_med':np.median(psf_per_chan[0]['BMIN']), 
                    'bpa_med':np.median(psf_per_chan[0]['BPA'])}
        bmaj_w = np.sum([np.median(psf_per_chan[i]['BMAJ'])*pb_values[i] for i in range(len(psf_per_chan))]) / np.sum(pb_values)
        bmin_w = np.sum([np.median(psf_per_chan[i]['BMIN'])*pb_values[i] for i in range(len(psf_per_chan))]) / np.sum(pb_values)
        bpa_w = np.sum([np.median(psf_per_chan[i]['BPA'])*pb_values[i] for i in range(len(psf_per_chan))]) / np.sum(pb_values)
        psf_weighted ={'beams':','.join(str(x) for x in ptgs2['beam']),'weights':','.join(str(x) for x in pb_values),
                    'bmaj_wghtd':bmaj_w, 'bmin_wghtd':bmin_w, 'bpa_wghtd':bpa_w}
        psf_field_median = get_psf_per_field(field, cube)
        
        # Also include the three nearest beams
        b=0
        n_beams = 3
        bmajs = np.zeros(n_beams) - 99
        bmins = np.zeros(n_beams) - 99
        bpas = np.zeros(n_beams) - 99
        while (b < len(bmajs)) and (b < len(ptgs2)):
            if os.path.isfile(psf_cubes[b]):
                bmajs[b] = np.median(psf_per_chan[b]['BMAJ'])
                bmins[b] = np.median(psf_per_chan[b]['BMIN'])
                bpas[b] = np.median(psf_per_chan[b]['BPA'])
            b += 1
        psf_three_nearest = {'bmajs':','.join("{:.7f}".format(x) for x in bmajs), 'bmins':','.join("{:.7f}".format(x) for x in bmins), 
                             'bpas':','.join("{:.7f}".format(x) for x in bpas)}

    else:
        print('This position has not been observed: {}'.format(source['name'][0]))
        forgotten_beams.append([source['name'][0], 'This position has not been observed', 99])
        psf_nearest = None
        psf_weighted = None
        psf_field_median = None
        psf_three_nearest = None
        # sys.exit()

    return psf_nearest, psf_weighted, psf_field_median, forgotten_beams, psf_three_nearest


if __name__ == '__main__':
    args = parse_args()
    package_dir = os.path.dirname(__file__)

    cube = args.cube
    catalog_file = args.catalog_file
    field = catalog_file.split('/')[-1].split('_')[0] 
    cat = Table.read(catalog_file, format='ascii', header_start=18)

    pointings = Table.read(package_dir + '/../data/pointings.dat', format='ascii.basic', delimiter=' ')
    pointings.rename_column('0taskID','taskID')
    obscensus = Table.read(package_dir + '/../data/obscensus.csv', comment='#')

    obs = obscensus[obscensus['name'] == field]
    ptgs = join(obs, pointings, keys='taskID', join_type='left')
    ptgs = unique(ptgs, keys=['name','beam'])

    out_catalog = []
    i = 0
    forgotten_beams_all = []

    # for s in cat[391:395]:
    for s in cat:
        psf_dict={'name':s['name']}
        field = catalog_file.split('/')[0].split('_')[-1]

        psf_nearest, psf_weighted, psf_field_median, forgotten_beams, psf_3nearest= main(s, field, cube, ptgs, pb_root_dir=args.pb_root_dir)
        if psf_nearest != None:
            psf_dict.update(psf_nearest)
            psf_dict.update(psf_weighted)
            psf_dict.update(psf_field_median)
            psf_dict.update(psf_3nearest)
            out_catalog.append(psf_dict)
        if len(forgotten_beams) > 0:
            forgotten_beams_all.append(forgotten_beams)
        i += 1

    forgotten_beams_all = np.array(forgotten_beams_all).reshape(-1,3)
    out_name = str(args.catalog_file)[:-4]+'_forgotten_beams.txt'
    tab_forgotten = Table(forgotten_beams_all,names=['name','forgotten_psf','psf_rank'])
    if len(forgotten_beams_all) > 0:
        tab_forgotten.write(out_name, format='ascii.fixed_width', delimiter=' ', overwrite=True)
        print("\tSaved 'forgotten' beams table to {}".format(out_name))

    out_table = Table(out_catalog)
    out_name = str(args.catalog_file)[:-4]+'_beams.txt'
    out_table.write(out_name, format='ascii.fixed_width', delimiter=' ', overwrite=True)
    print("\tSaved output table to {}".format(out_name))
