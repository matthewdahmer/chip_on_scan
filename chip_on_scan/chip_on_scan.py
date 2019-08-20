# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from urllib.request import urlopen
from urllib.error import URLError
import json
from os.path import expanduser
from itertools import product
from copy import deepcopy

from Chandra.Time import DateTime
import xija
from kadi.commands import states
from Chandra.cmd_states import reduce_states


def load_model_specs(check_internet=True):
    """ Load Xija model parameters for all available models.

    Returns:
        dictionary: A dictionary containing the model specifications for all available Xija models

    Note:
        This will need to be updated as new models are approved or existing models are renamed.

    """

    home = expanduser("~")

    def get_model(branch, internet):
        """ Load parameters for a single Xija model.

        Args:
            branch (str): Relative location of model file, starting from the chandra_models/chandra_models/xija/
                directory
            internet (bool): Availability of an internet connection, for accessing github.com

        Returns:
            dictionary: JSON file stored as a dictionary, containing Xija model parameters

        """

        url = 'https://raw.githubusercontent.com/sot/chandra_models/master/chandra_models/xija/'
        local_dir = '/AXAFLIB/chandra_models/chandra_models/xija/'

        if internet:
            model_spec_url = url + branch  # aca/aca_spec.json'
            with urlopen(model_spec_url) as url:
                response = url.read()
                f = response.decode('utf-8')
        else:
            with open(home + local_dir + branch) as fid:  # 'aca/aca_spec.json', 'rb') as fid:
                f = fid.read()
        return json.loads(f)

    model_specs = {}

    internet = False
    if check_internet:
        try:
            _ = urlopen('https://github.com')
            internet = True
        except URLError:
            internet = False

    model_specs['aacccdpt'] = get_model('aca/aca_spec.json', internet)
    model_specs['1deamzt'] = get_model('dea/dea_spec.json', internet)
    model_specs['1dpamzt'] = get_model('dpa/dpa_spec.json', internet)
    model_specs['fptemp'] = get_model('acisfp/acisfp_spec.json', internet)
    model_specs['1pdeaat'] = get_model('psmc/psmc_spec.json', internet)
    model_specs['pftank2t'] = get_model('pftank2t/pftank2t_spec.json', internet)
    model_specs['tcylaft6'] = get_model('tcylaft6/tcylaft6_spec.json', internet)
    model_specs['4rt700t'] = get_model('fwdblkhd/4rt700t_spec.json', internet)
    model_specs['pline03t'] = get_model('pline/pline03t_model_spec.json', internet)
    model_specs['pline04t'] = get_model('pline/pline04t_model_spec.json', internet)

    return model_specs


def c_to_f(temp):
    """ Convert Celsius to Fahrenheit

    Args:
        temp (int, float, numpy.ndarray, list, tuple): Temperature in Celsius

    Returns:
        (int, float, numpy.ndarray, list, tuple): Temperature in Fahrenheit

    """
    if type(temp) is list or type(temp) is tuple:
        return [c * 1.8 + 32 for c in temp]
    else:
        return temp * 1.8 + 32.0


def f_to_c(temp):
    """ Convert Fahrenheit to Celsius

    Args:
        temp (int, float, numpy.ndarray, list, tuple): Temperature in Fahrenheit

    Returns:
        (int, float, numpy.ndarray, list, tuple): Temperature in Celsius

    """
    if type(temp) is list or type(temp) is tuple:
        return [(c - 32) / 1.8 for c in temp]
    else:
        return (temp - 32.0) / 1.8


def setup_model(msid, t0, t1, model_spec, init):
    """ Create Xija model object

    This function creates a Xija model object with initial parameters, if any. This function is intended to create a
    streamlined method to creating Xija models that can take both single value data and time defined data
    (e.g. [pitch1, pitch2, pitch3], [time1, time2, time3]), defined in the `init` dictionary.

    Args:
        msid (str): Primary MSID for model; in this case it can be anything as it is only being used to name the model,
            however keeping the convention to name the model after the primary MSID being predicted reduces confusion
        t0 (str, float, int): Start time for model prediction; this can be any format that Chandra.Time.DateTime accepts
        t1 (str, float, int): End time for model prediction; this can be any format that Chandra.Time.DateTime accepts
        model_spec (dict, string): Dictionary of model parameters or file location where parameters can be imported
        init (dict): Dictionary of Xija model initialization parameters, can be empty

    Returns:
        (xija.model.XijaModel): Xija model object

    Example:

        model_specs = load_model_specs()

        init = {'1dpamzt': 35., 'dpa0': 35., 'eclipse': False, 'roll': 0, 'vid_board': True, 'pitch':155,
                          'clocking': True, 'fep_count': 5, 'ccd_count': 5, 'sim_z': 100000}

        model = setup_model('1dpamzt', '2019:001:00:00:00', '2019:010:00:00:00', model_specs['1dpamzt'], init)

    Notes:
        This does not run the model, only sets up the model to be run.

        Any parameters not specified in `init` will either need to be pulled from telemetry or explicitly defined
        outside of this function before running the model.

    """

    model = xija.ThermalModel(msid, start=t0, stop=t1, model_spec=model_spec)
    for key, value in init.items():
        if isinstance(value, dict):
            model.comp[key].set_data(value['data'], value['times'])
        else:
            model.comp[key].set_data(value)

    return model


def run_profile(times, schedule, msid, model_spec, init, pseudo=None):
    """ Run a Xija model for a given time and state profile.

    Args:
        times (numpy.ndarray): Array of time values, in seconds from '1997:365:23:58:56.816' (Chandra.Time.DateTime
            epoch)
        schedule (dict): Dictionary of pitch, roll, etc. values that match the time values specified above in `times`
        msid (str): Primary MSID for model being run
        model_spec (dict, string): Dictionary of model parameters or file location where parameters can be imported
        init (dict): Dictionary of Xija model initialization parameters, can be empty but not recommended
        pseudo (:obj:`str`, optional): Name of one or more pseudo MSIDs used in the model, if any, only necessary if one
            wishes to retrieve model results for this pseudo node, if it exists

    Returns:
        dict: Dictionary of results, keys are node names (e.g. 'aacccdpt', 'aca0'), values are Xija model component
            objects

    Example:

        times = np.array(DateTime(['2019:001:00:00:00', '2019:001:12:00:00', '2019:002:00:00:00',
                                   '2019:003:00:00:00']).secs)

        pitch = np.array([150, 90, 156, 156])

        schedule = {'pitch':pitch}

        model_specs = load_model_specs()

        init = {'1dpamzt': 20., 'dpa0': 20., 'eclipse': False, 'roll': 0, 'vid_board': True,
                          'clocking': True, 'fep_count': 5, 'ccd_count': 5, 'sim_z': 100000}

        results = run_profile(times, schedule, '1dpamzt', model_specs['1dpamzt'], init, pseudo='dpa0')

    Notes:
        Any parameters specified in `init` will be overwritten by those specified in the body of this function, if they
        happen to be defined in both places.

    """

    model = setup_model(msid, times[0], times[-1], model_spec, init)

    for key, value in schedule.items():
        model.comp[key].set_data(value, times=times)

    model.make()
    model.calc()
    tmsid = model.get_comp(msid)
    results = {msid: tmsid}

    if pseudo is not None:
        results[pseudo] = model.get_comp(pseudo)

    return results


def test_run_profile(dpa_model_spec, dpa_init):
    """ Test for the run_profile function.

    Args:
        dpa_model_spec (dict): DPA model specification in dictionary form (read from json file)

    Returns:
        None

    """
    test_times = np.array(
        DateTime(['2019:001:00:00:00', '2019:001:12:00:00', '2019:002:00:00:00', '2019:003:00:00:00']).secs)
    pitch = np.array([150, 90, 156, 156])
    roll = np.array([0, -5, 10, 0])
    test_schedule = {'pitch': pitch, 'roll': roll}
    results = run_profile(test_times, test_schedule, '1dpamzt', dpa_model_spec, dpa_init)
    dpa_results = results['1dpamzt']
    assert np.all(dpa_results.mvals > -10)
    assert len(dpa_results.times) > 1


def get_state_data(tstart, tstop):
    """ Get states where 'vid_board', 'clocking', 'fep_count', 'pcad_mode' are constant.

    Args:
        tstart (int, float, string): Start time, using Chandra.Time epoch
        tstop (int, float, string): Stop time, using Chandra.Time epoch

    Returns:
        (numpy.ndarray): state data

    """
    keys = ['pitch', 'off_nom_roll', 'ccd_count', 'fep_count', 'clocking', 'vid_board', 'pcad_mode', 'simpos']
    state_data = states.get_states(tstart, tstop, state_keys=keys, merge_identical=True)

    # Convert 'trans_keys' elements from 'TransKeysSet' objects to strings for compatibility with the 'reduce_states'
    # function in the 'Chandra.cmd_states' package
    state_data['trans_keys'] = [str(val) for val in state_data['trans_keys']]
    state_data['tstart'] = DateTime(state_data['datestart']).secs  # Add start time in seconds as 'tstart'
    state_data['tstop'] = DateTime(state_data['datestop']).secs  # Add stop time in seconds as 'tstop'

    # relying on 'pcad_mode' to ensure attitude does not change significantly within a dwell
    state_data = reduce_states(state_data, ['ccd_count', 'fep_count', 'clocking', 'vid_board', 'pcad_mode'])

    return state_data


def get_schedule_dict(state_data):
    """ Convert state data to a dictionary formatted for the `run_profile` function.

    Args:
        state_data (numpy.ndarray): Structured array of state data output by `cmd_states` or `get_state_data`

    Returns:
        times: Numpy ndarray of time values matching each item in the returned schedule dictionary
        dict: Dictionary of results, keys are node names (e.g. 'aacccdpt', 'aca0'), values are Xija model component
            objects

    Notes:
        The returned dictionary is in a form that makes it very straighforward to apply to a Xija model

    """

    times = np.array(list((zip(state_data['tstart'], state_data['tstop'])))).reshape((-1,))
    pitch = np.array(list((zip(state_data['pitch'], state_data['pitch'])))).reshape((-1,))
    roll = np.array(list((zip(state_data['off_nom_roll'], state_data['off_nom_roll'])))).reshape((-1,))
    clocking = np.array(list((zip(state_data['clocking'], state_data['clocking'])))).reshape((-1,))
    vid_board = np.array(list((zip(state_data['vid_board'], state_data['vid_board'])))).reshape((-1,))
    ccd_count = np.array(list((zip(state_data['ccd_count'], state_data['ccd_count'])))).reshape((-1,))
    fep_count = np.array(list((zip(state_data['fep_count'], state_data['fep_count'])))).reshape((-1,))
    sim_z = np.array(list((zip(state_data['simpos'], state_data['simpos'])))).reshape((-1,))

    schedule = {'pitch': pitch,
                'roll': roll,
                'fep_count': fep_count,
                'ccd_count': ccd_count,
                'clocking': clocking,
                'vid_board': vid_board,
                'sim_z': sim_z}

    return times, schedule


def get_acis_fp_limits(states, acis_s=-111.0, acis_i=-112.0):
    """ Get ACIS Focal Plane limits based on SIM position.

    This function uses SIM position to assign temperature limits.

    Args:
        states (ndarray): States array, in the form returned by the cmd_states.get_cmd_states.fetch_states function
        acis_s (float, optional): Limit for ACIS S observations
        acis_i (float, optional): Limit for ACIS I observations

    Returns:
        (ndarray): Array of ACIS limits that match the states array

    """

    states_acis_limits = np.zeros(len(states)) + 20  # Set limit initially to 20C
    ind_acis_s = (states['simpos'] > 0) & (states['simpos'] < 80669) & (states['clocking'] == 1) & (
                states['vid_board'] == 1)
    ind_acis_i = (states['simpos'] > 83826) & (states['clocking'] == 1) & (states['vid_board'] == 1)
    states_acis_limits[ind_acis_s] = acis_s
    states_acis_limits[ind_acis_i] = acis_i

    return states_acis_limits


def get_acis_limits(states):
    """ Get ACIS DPA, DEA, PSMC limits.

    This function is intended to be temporary, to facilitate long term comparison of additional chip opportunities in
    past schedules.

    Args:
        states (ndarray): States array, in the form returned by the cmd_states.get_cmd_states.fetch_states function

    Returns:
        (dict): Dictionary of ndarrays, each returning ACIS limits that match the states array for the specified
                msid/key

    """

    states_acis_fp_limits = get_acis_fp_limits(states, acis_s=-111.0, acis_i=-112.0)
    ind_old = states['tstart'] <= DateTime('2019:172:00:00:00').secs

    states_acis_fp_limits[(states_acis_fp_limits == -112.0) & ind_old] = -114.0
    states_acis_fp_limits[(states_acis_fp_limits == -111.0) & ind_old] = -112.0

    states_acis_dpa_limits = np.zeros(len(states))
    ind_old = states['tstart'] <= DateTime('2019:172:00:00:00').secs
    ind_new = states['tstart'] > DateTime('2019:172:00:00:00').secs

    states_acis_dpa_limits[ind_old] = 36.5
    states_acis_dpa_limits[ind_new] = 37.5

    states_acis_dea_limits = np.zeros(len(states)) + 35.5

    states_acis_psmc_limits = np.zeros(len(states)) + 52.5

    limits = {'fptemp': states_acis_fp_limits,
              '1dpamzt': states_acis_dpa_limits,
              '1deamzt': states_acis_dea_limits,
              '1pdeaat': states_acis_psmc_limits}

    return limits


def get_max_dwell_mvals(model, state_data):
    """ Get maximum data for each dwell as defined in state_data in the Xija model object `model`.

    Args:
        model (Xija object): Xija model object for a given MSID (i.e. xija_model.get_comp('1dpamzt'))
        state_data (ndarray): States array, in the form returned by cmd_states.get_cmd_states.fetch_states


    Returns:
        (tuple): Maximum model values within each dwell defined by the input states array

    """
    dwell_results = []
    for ind in range(len(state_data)):
        ind_dwell = (model.times >= state_data['tstart'][ind]) & (model.times <= state_data['tstop'][ind])
        if np.any(ind_dwell):
            dwell_results.append(np.max(model.mvals[ind_dwell]))
        else:
            dwell_results.append(-1.0e6)
    return tuple(dwell_results)


# def evaluate_all_cases_for_one_msid(state_limits, all_case_results):
#     """ Evaluate all case permutations for one model/MSID
#
#     Args:
#         state_limits (ndarray): Limits for each state
#         all_case_results (list, tuple): List of case results, with each element representing the output from
#             `get_max_dwell_mvals`
#
#     Returns:
#         (ndarray): Boolean array representing whether or not data in each case fall within limits
#
#     """
#     all_case_results = np.array(all_case_results)
#     n_cases = np.shape(all_case_results)[0]
#
#     # The first case has zero changes, look for cases that chage more than 0.1 degrees (refine later).
#     msid_inc_ind = (all_case_results - all_case_results[0]) > 0.1
#     msid_limit_array = np.tile(state_limits, (n_cases, 1))
#     msid_ok = np.zeros(np.shape(all_case_results)) < 1
#     msid_bad = msid_inc_ind & (all_case_results > msid_limit_array)
#     msid_ok[msid_bad] = False
#     return msid_ok


def evaluate_one_case_for_one_msid(state_limits, baseline_case_results, case_results, debug=True):
    """ Evaluate one case for one model/MSID

    Args:
        state_limits (ndarray): Limits for each state
        baseline_case_results (list, tuple): Results for baseline case, representing the output from
            `get_max_dwell_mvals`
        case_results (list, tuple): Results for one case, representing the output from `get_max_dwell_mvals`
    Returns:
        (ndarray): Boolean array representing whether or not data in each case fall within limits

    """

    if debug:
        state_limits = state_limits

    # The first case has zero changes, look for cases that chage more than 0.1 degrees (refine later).
    msid_inc_ind = np.array([c - b for c, b in zip(case_results, baseline_case_results)]) > 0.1
    msid_ok = np.zeros(len(case_results)) < 1

    msid_bad = msid_inc_ind & (case_results > state_limits)
    msid_ok[msid_bad] = False
    return msid_ok


def find_modifiable_states(state_data):
    """ Find indices into the state_data array,

    Args:
        state_data (ndarray): States array, in the form returned by cmd_states.get_cmd_states.fetch_states

    Returns:
        (ndarray): Numeric index of states that represent dwells with modifiable chip counts
        (list): List of all permutations, where each row represents a single case (combination), and each column
            represents the number of chips to be added for each state indexed by the returned ndarray (above)

    Note:
        The first element in the returned list represents 0's for each modifiable state, representing the baseline case.

    """

    modifiable = (state_data['pcad_mode'] == 'NPNT') & (state_data['clocking'] == 1) & (
                state_data['fep_count'] == state_data['ccd_count']) & (state_data['fep_count'] < 4)
    states_ind = np.where(modifiable)[0]
    cases = list(product([0, 1], repeat=len(states_ind)))
    return states_ind, cases


def run_cases(state_data, modifiable_states_ind, acis_state_limits, cases, times, schedule):
    """ Run all valid chip turn-on permutations

    Args:
    state_data (ndarray): States array, in the form returned by cmd_states.get_cmd_states.fetch_states
        modifiable_states_ind (ndarray): Numeric index of states that represent dwells with modifiable chip counts
        acis_state_limits (dict): Dictionary of ndarrays, each returning ACIS limits that match the states array for
            the specified msid/key
        cases (list): List of all permutations, where each row represents a single case (combination), and each
            column represents the number of chips to be added for each state indexed by `modifiable_states_ind`. The
            first case is assumed to be the baseline case.
        times (numpy.ndarray): Array of time values, in seconds from '1997:365:23:58:56.816' (Chandra.Time.DateTime
            epoch)
        schedule (dict): Dictionary of pitch, roll, etc. values that match the time values specified above in `times`

    Returns:
        (dict): Dictionary of case results, where each item in the dictionary is an ndarray of length `len(cases)`

    """
    all_dpa_case_results = {}
    all_dea_case_results = {}
    all_psmc_case_results = {}
    all_fp_case_results = {}

    dpa_diagnostic_results = {}
    dea_diagnostic_results = {}
    psmc_diagnostic_results = {}
    fp_diagnostic_results = {}

    all_dpa_ok = {}
    all_dea_ok = {}
    all_psmc_ok = {}
    all_fp_ok = {}

    n = -1
    loop_cases = deepcopy(cases)
    zero_case = cases[0]

    max_cases = len(cases)

    while len(loop_cases) > 0:
        n = n + 1
        case = loop_cases.pop(0)
        if np.mod(n, 10) == 0:
            print('Running case {} out of {}'.format(n + 1, max_cases))

        # Generate new schedule data for CCD and FEP count
        mod_states_ccd_count = deepcopy(state_data['ccd_count'])
        mod_states_ccd_count[modifiable_states_ind] = mod_states_ccd_count[modifiable_states_ind] + np.array(case)
        ccd_count = np.array(list((zip(mod_states_ccd_count, mod_states_ccd_count)))).reshape((-1))

        mod_states_fep_count = deepcopy(state_data['fep_count'])
        mod_states_fep_count[modifiable_states_ind] = mod_states_fep_count[modifiable_states_ind] + np.array(case)
        fep_count = np.array(list((zip(mod_states_fep_count, mod_states_fep_count)))).reshape((-1))

        schedule['fep_count'] = fep_count
        schedule['ccd_count'] = ccd_count

        # Run the new profile
        dpa_case_results = run_profile(times, schedule, '1dpamzt', model_specs['1dpamzt'], model_init['1dpamzt'])
        dea_case_results = run_profile(times, schedule, '1deamzt', model_specs['1deamzt'], model_init['1deamzt'])
        psmc_case_results = run_profile(times, schedule, '1pdeaat', model_specs['1pdeaat'], model_init['1pdeaat'])
        fp_case_results = run_profile(times, schedule, 'fptemp', model_specs['fptemp'], model_init['fptemp'])

        # Determine the maxiumum temperatures for this case
        max_dpa = get_max_dwell_mvals(dpa_case_results['1dpamzt'], state_data)
        max_dea = get_max_dwell_mvals(dea_case_results['1deamzt'], state_data)
        max_psmc = get_max_dwell_mvals(psmc_case_results['1pdeaat'], state_data)
        max_fp = get_max_dwell_mvals(fp_case_results['fptemp'], state_data)

        # Store these cases (will delete later if bad)
        all_dpa_case_results[case] = max_dpa
        all_dea_case_results[case] = max_dea
        all_psmc_case_results[case] = max_psmc
        all_fp_case_results[case] = max_fp

        # Evaluate the current case against all models
        dpa_ok = evaluate_one_case_for_one_msid(acis_state_limits['1dpamzt'], all_dpa_case_results[zero_case],
                                                max_dpa)
        dea_ok = evaluate_one_case_for_one_msid(acis_state_limits['1deamzt'], all_dea_case_results[zero_case],
                                                max_dea)
        psmc_ok = evaluate_one_case_for_one_msid(acis_state_limits['1pdeaat'], all_psmc_case_results[zero_case],
                                                 max_psmc)
        fp_ok = evaluate_one_case_for_one_msid(acis_state_limits['fptemp'], all_fp_case_results[zero_case],
                                               max_fp)
        all_ok = dpa_ok & dea_ok & psmc_ok & fp_ok

        if not np.all(all_ok):
            print('Case {} is bad'.format(case))
            all_dpa_case_results.pop(case)
            all_dea_case_results.pop(case)
            all_psmc_case_results.pop(case)
            all_fp_case_results.pop(case)

            first_change = case.index(1)

            if all_ok[modifiable_states_ind[first_change]] is not True:
                # Eliminate all other cases that use the failing case
                original_len = len(loop_cases)
                loop_cases = [c for c in loop_cases if c[first_change] != 1]
                new_len = len(loop_cases)
                max_cases = max_cases - (original_len - new_len)

        else:
            all_dpa_ok[case] = dpa_ok
            all_dea_ok[case] = dea_ok
            all_psmc_ok[case] = psmc_ok
            all_fp_ok[case] = fp_ok

        # Store results for later inspection
        dpa_diagnostic_results[case] = {'times': dpa_case_results['1dpamzt'].times,
                                        'mvals': dpa_case_results['1dpamzt'].mvals}
        dea_diagnostic_results[case] = {'times': dea_case_results['1deamzt'].times,
                                        'mvals': dea_case_results['1deamzt'].mvals}
        psmc_diagnostic_results[case] = {'times': psmc_case_results['1pdeaat'].times,
                                         'mvals': psmc_case_results['1pdeaat'].mvals}
        fp_diagnostic_results[case] = {'times': fp_case_results['fptemp'].times,
                                       'mvals': fp_case_results['fptemp'].mvals}

    diagnostic_results = {'1dpamzt': dpa_diagnostic_results,
                          '1deamzt': dea_diagnostic_results,
                          '1pdeaat': psmc_diagnostic_results,
                          'fptemp': fp_diagnostic_results}

    case_results = {'1dpamzt': all_dpa_ok,
                    '1deamzt': all_dea_ok,
                    '1pdeaat': all_psmc_ok,
                    'fptemp': all_fp_ok,
                    'ok_cases': all_dpa_ok.keys()} # Only OK cases are kept for all models

    return case_results, diagnostic_results


if __name__ == "__main__":

    model_specs = load_model_specs()

    model_init = {'1dpamzt': {'1dpamzt': 20., 'dpa0': 20., 'eclipse': False, 'roll': 0, 'vid_board': True,
                              'clocking': True, 'fep_count': 5, 'ccd_count': 5, 'sim_z': 100000, 'dpa_power': 0.0},
                  '1deamzt': {'1deamzt': 20., 'eclipse': False, 'roll': 0, 'vid_board': True,'clocking': True,
                              'fep_count': 5, 'ccd_count': 5, 'sim_z': 100000, 'dpa_power': 0.0},
                  '1pdeaat': {'1pdeaat': 30., 'pin1at': 20., 'eclipse': False, 'roll': 0, 'vid_board': True,
                              'clocking': True, 'fep_count': 5, 'ccd_count': 5, 'sim_z': 100000, 'dpa_power': 0.0,
                              'dh_heater':0},
                  'fptemp': {'fptemp': -112., '1cbat': -55., 'sim_px': -110., 'eclipse': False, 'dpa_power': 0.0,
                             'sim_z': 100000, 'orbitephem0_x': 100000e3, 'orbitephem0_y': 100000e3,
                             'orbitephem0_z': 100000e3, 'aoattqt1': 0.0, 'aoattqt2': 0.0, 'aoattqt3': 0.0,
                             'aoattqt4': 1.0, 'dh_heater': False}}

    tstart = '2019:100:00:00:00'
    tstop = '2019:120:12:00:00'

    timethis1 = DateTime().secs
    state_data = get_state_data(tstart, tstop)
    acis_state_limits = get_acis_limits(state_data)
    times, schedule = get_schedule_dict(state_data)

    modifiable_states_ind, cases = find_modifiable_states(state_data)

    case_results, diagnostic_results = run_cases(state_data, modifiable_states_ind, acis_state_limits, cases, times,
                                                 schedule)

    print('The following cases passed: {}'.format(case_results['ok_cases']))

    timethis2 = DateTime().secs
    print('This took {} seconds for {} cases (2^{})'.format(timethis2 - timethis1, len(cases),
                                                            len(modifiable_states_ind)))

    print(f'State indices: {modifiable_states_ind}')
