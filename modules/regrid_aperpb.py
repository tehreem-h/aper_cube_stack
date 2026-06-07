from argparse import ArgumentParser, RawTextHelpFormatter
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
import numpy as np
from reproject import reproject_interp


###################################################################

def parse_args():
    parser = ArgumentParser(description="Make plots of sourcefinding.py results.",
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('-t', '--template_image', default=None,
                        help='Specify the input template image (default: %(default)s).')

    parser.add_argument('-b', '--beams', default='0-39',
                        help='Specify a range (0-39) or list (3,5,7,11) of beams on which to do source finding (default: %(default)s).')

    parser.add_argument('-p', '--pb_root_dir', default=None,
                        help='Specify the root directory where the primary beam models are located (default: %(default)s).')

    parser.add_argument('-o', '--output', default=None,
                        help='Specify the output (default: appends "_pb.fits" to input).')

    # Parse the arguments above
    arguments = parser.parse_args()
    return arguments


def get_cb_model_freq():
    """
    Set the central frequency for the Gaussian regression beams based on the Apertif DR1 documentation.
    """
    alexander_orig_dr1 = 1361.25 * u.MHz
    return alexander_orig_dr1


def model_lookup(beam, root_dir=''):
    """
    Find appropriate beam model from Gaussian regression method.
    For now, does not search as a function of time.
    """
    model = root_dir + '/{:02}_gp_avg_orig.fits'.format(beam)
    return model


def main(template_im, beam, pb_root_dir='', output=''):

    pb_model = model_lookup(beam, root_dir=pb_root_dir)
    temp_header = fits.getheader(template_im)

    # Change the reference pixel of beam model to reference pixel of image to correct
    hdulist_pb = fits.open(pb_model)
    hdulist_pb[0].header['CRVAL1'] = temp_header['CRVAL1']
    hdulist_pb[0].header['CRVAL2'] = temp_header['CRVAL2']

    # Rescale to appropriate frequency
    avg_cube_freq = (temp_header['CRVAL3'] + temp_header['CDELT3'] * temp_header['NAXIS3']) * u.Hz
    hdulist_pb[0].header['CDELT1'] = (
                hdulist_pb[0].header['CDELT1'] * get_cb_model_freq().to(u.Hz) / avg_cube_freq).value
    hdulist_pb[0].header['CDELT2'] = (
                hdulist_pb[0].header['CDELT2'] * get_cb_model_freq().to(u.Hz) / avg_cube_freq).value

    print("\tRegridding Gaussian regression primary beam for beam {} to {}".format(beam, template_im))
    cb_reprojected, footprint = reproject_interp(hdulist_pb, WCS(temp_header).celestial, [temp_header['NAXIS2'],
                                                                                          temp_header['NAXIS2']])
    d_new = np.ones((temp_header['NAXIS3'], temp_header['NAXIS2'], temp_header['NAXIS2']))
    d_beam_cube = d_new * cb_reprojected
    cb_reprojected = np.float32(d_beam_cube)
    hdulist_pb.close()

    hdu = fits.PrimaryHDU(data=cb_reprojected, header=temp_header)

    if (not output) and ('_image.fits' in template_im):
        output = template_im[:-11] + '_pb.fits'
    elif (not output):
        output = template_im[:-5] + '_pb.fits'

    hdu.writeto(output, overwrite=True)

    return


if __name__ == '__main__':
    args = parse_args()
    # Range of beams to work on:
    if '-' in args.beams:
        b_range = args.beams.split('-')
        beams = np.array(range(int(b_range[1])-int(b_range[0])+1)) + int(b_range[0])
    else:
        beams = [int(b) for b in args.beams.split(',')]

    for b in beams:
        main(args.template_image, b, pb_root_dir=args.pb_root_dir, output=args.output)
