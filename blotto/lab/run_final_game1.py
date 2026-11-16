"""Final frozen Game 1 challenge, with known-evidence selection and fresh tests."""
from fractions import Fraction
from pathlib import Path
import argparse
import json
import math
import platform
import shutil
import sys
import time
import numpy as np
from blotto_lab.core import Population,best_response,mix_tables,security_of
from blotto_lab.contamination import FrontierObjective,exact_counter,search_frontiers
from blotto_lab.historical import rows_to_table,sampled_metadata
from final_tools import final_definitions,final_rows,enhanced_path,population_from_shares
from run_contamination_study import nominal_winner
from run_multiwave_study import ROOT,unpack_table
from run_population_study import save,sha,write_csv
from verify_population_study import rational


def read(path):return json.loads(path.read_text())


def pack(t):return {'table':{'numerators':t.numerators,'denominator':t.denominator,'evidence':t.evidence}}


def prepare_selection(merged,diagnostic,defs,cfg):
    base=unpack_table(defs['tables'][defs['search_keys'][defs['midpoint']['table_index']]],'known midpoint')
    keys=list(diagnostic['critical_scenarios']);cases=[unpack_table(merged['models'][k],k)for k in keys]
    tables=[base];names=['midpoint'];objectives=[];groups={}
    for raw in cfg['replacement_caps']:
        cap=Fraction(raw);indices=[0]
        for k,t in zip(keys,cases):
            indices.append(len(tables));tables.append(mix_tables([base,t],[1-cap,cap]));names.append('cap'+raw+'__'+k)
        objectives.append(FrontierObjective('replacement_'+raw+'_regret',tuple(indices),'regret'))
    for k,t in zip(keys,cases):groups.setdefault(merged['models'][k]['kind'],[]).append(t)
    balanced=mix_tables([mix_tables(v,[1]*len(v))for v in groups.values()],[1]*len(groups))
    hedge=Fraction(cfg['mean_hedge_share']);tables.append(mix_tables([base,balanced],[1-hedge,hedge]));names.append('balanced_failure_hedge')
    objectives.append(FrontierObjective('balanced_failure_hedge',(len(tables)-1,),'score'))
    return tables,names,objectives,keys


def evaluate(candidates,tables,models,families):
    scores={};regrets={};summary={}
    optimum={k:best_response(t).expected_score for k,t in tables.items()}
    for cid,c in candidates.items():
        x=c['allocation'];scores[cid]={k:t.expected_score(x)for k,t in tables.items()}
        regrets[cid]={k:optimum[k]-v for k,v in scores[cid].items()};summary[cid]={}
        for phase in ['challenge','confirmation']:
            individual=[k for k,m in models.items()if m['phase']==phase and m['kind']in ['fresh_static','fresh_learning']]
            family_keys=list(families[phase].values());fs=[scores[cid][k]for k in family_keys]
            minimum=min(scores[cid][k]for k in individual);regret=max(regrets[cid][k]for k in individual)
            summary[cid][phase]={'family_balanced_mean':sum(fs)/len(fs),'family_minimum':min(fs),
                 'individual_minimum':minimum,'individual_worst_regret':regret,
                 'minimum_scenarios':[k for k in individual if scores[cid][k]==minimum],
                 'worst_regret_scenarios':[k for k in individual if regrets[cid][k]==regret]}
    return scores,regrets,summary


def figures(out,r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(r['primary']);fig,axes=plt.subplots(1,2,figsize=(13,6.5),sharey=True)
    ypos=np.arange(len(names));width=.36
    for j,phase in enumerate(['challenge','confirmation']):
        means=[float(r['summary'][r['primary'][n]['candidate_id']][phase]['family_balanced_mean'])for n in names]
        floors=[float(r['summary'][r['primary'][n]['candidate_id']][phase]['family_minimum'])for n in names]
        axes[j].barh(ypos-width/2,means,width,label='Eight-family mean',color='#267ba5')
        axes[j].barh(ypos+width/2,floors,width,label='Lowest family mean',color='#e2a24a')
        axes[j].axvline(7.5,color='#555555',ls=':',lw=1);axes[j].set_xlim(0,10)
        axes[j].set_title(phase.title());axes[j].set_xlabel('Official expected points')
        axes[j].set_yticks(ypos,[n.replace('_',' ')for n in names],fontsize=8);axes[j].invert_yaxis();axes[j].grid(axis='x',alpha=.15)
    axes[1].legend(fontsize=8);fig.suptitle('Frozen primary rules on new constructed populations\nDeclared family weights; no applicant-frequency estimate')
    fig.tight_layout();fig.savefig(out/'primary-comparison.png',dpi=170);plt.close(fig)


def report(out,r):
    lines=['**Final Game 1 challenge: frozen selections on new populations**','',
        f"The study froze {len(r['candidates'])} distinct candidates and {len(r['primary'])} primary comparison "
        'rules before generating any evaluation rows or running fresh arenas. All older outcomes were '
        'treated as known development evidence. Neither phase changes the candidates. No final '
        'application vector is selected here.','',
        '**Primary comparisons**','',
        'Each phase uses eight equally weighted family means: four coalition-law families, four-field '
        'priorities, all-field diffuse allocation, arena final snapshots, and arena time averages. '
        'Those weights are an explicit convention. The minimum family score and individual worst '
        'regret answer different robustness questions.','',
        '| Frozen rule | Allocation | Challenge mean / family minimum / worst individual regret | Confirmation mean / family minimum / worst individual regret | Pure guarantee |',
        '| --- | --- | ---: | ---: | ---: |']
    for name,p in r['primary'].items():
        cid=p['candidate_id'];cells=[]
        for phase in ['challenge','confirmation']:
            s=r['summary'][cid][phase];cells.append(' / '.join(f'{float(s[k]):.3f}'for k in ['family_balanced_mean','family_minimum','individual_worst_regret']))
        lines.append(f"| {name} | {tuple(p['allocation'])} | {' | '.join(cells)} | {float(r['security'][cid]['guarantee']):g} |")
    lines+=['','![Frozen primary comparisons](primary-comparison.png)','',
        '**Fresh construction and evidence limits**','',
        'Each phase has 26 independent generator populations of 200,000 legal draws, six new retained '
        'strategy arenas, and final/time-average snapshots. The 52 direct populations therefore use '
        '10.4 million new legal joint vectors. The second phase shifts scouts, heaping, middle-five '
        'bids, and priority concentration. It also uses sparse zero-mutation arena starts. The phases '
        'have different laws; differences between their summaries are not a statistical train/test gap.','',
        'Six static-mixture diagnostics and six static/learner-mixture diagnostics per phase combine '
        'the new components using saved exact weights. Eight family tables are also saved per phase. '
        'These derived populations reuse evidence; they are excluded from the individual extrema and '
        'are not counted as independent sampled observations.','',
        '**Three new selection rules, fixed before the challenge**','',
        'Two rules minimize worst regret over the convex hull of the old midpoint population and '
        'populations that replace 10% or 25% with any mixture of the ten known failure cases. The '
        'third optimizes the midpoint with a 10% hedge toward a reference balanced first by evidence '
        'kind and then within kind. All require pure security at least two. None of those assumptions '
        'estimates actual applicant shares. All earlier merged primaries and the machine max-min '
        'baseline remain primary comparisons. The literal human vectors remain in the full archive.','',
        'The full legal action space is searched within exact security classes. The worst regret '
        'over each continuous replacement set occurs at a vertex because population regret is '
        'convex. All conditional optima and tie counts are saved. Independent integer upper bounds '
        'and outcome-pattern security verify the selections.','',
        '**Broader adaptation**','',
        'Each of the twelve arenas runs 1,200 steps with fixed point/win type shares of 0%, 50%, or '
        '100%, under replicator or payoff-replacement revision. Every 100 steps, exact point responses '
        'to the current and historical population are proposed, together with two large troop-transfer '
        'mutations. New actions are seeded in both types, and old counters remain available. Point '
        'and win objectives never compete by comparing their incompatible units. Win optimization '
        'remains restricted to available actions. There is no equilibrium-convergence claim.','',
        '**Precision, integrity, and the next decision**','',
        f"The simultaneous 95% DKW score-error bound is {r['sampling_bound']:.4f} points for direct "
        'generator populations, conditional on their laws. Convex mixtures inherit their component '
        'bounds. It does not validate the laws, population weights, or induced learning populations. '
        'All saved-table scores, regrets, and quantized-mixture certificates are exact rational values.','',
        'The independent audit checks all fresh draws, arena replay and innovations, checkpoint '
        'content, type proportions, source hashes, the freeze boundary, derived mixtures, all score '
        'cells, and global selection. It is saved separately in acceptance.json.','',
        'The next step is the selection review: compare these common results with previously saved '
        'sensitivity and guarantees, make the population/risk tradeoff explicit, and choose a fixed '
        'vector. A difference between legitimate objectives does not itself require another research '
        'cycle. Any fitting to this new challenge would make it known evidence and require a further '
        'fresh test. Game 2 remains separate.','',
        'Run `python3 run_final_game1.py --output results/final-new`, then '
        '`python3 verify_final_game1.py --study results/final-new`. Existing output folders are '
        'preserved. Interrupted matching runs can use `--resume`.','']
    (out/'report.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'config/final-game1.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/final-game1');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.output.resolve();cfg=read(args.config);clock=time.perf_counter()
    sources=[Path(__file__),ROOT/'final_tools.py',ROOT/'verify_final_game1.py',ROOT/'FINAL-GAME1-PROTOCOL.md',ROOT/'failure_tools.py',
             ROOT/'run_contamination_study.py',ROOT/'run_multiwave_study.py',ROOT/'run_population_study.py',
             ROOT/'verify_failure_analysis.py',ROOT/'verify_population_study.py',ROOT/'tests/test_final_game1.py']+sorted((ROOT/'blotto_lab').glob('*.py'))
    inputs=[ROOT/'results/merged-game1/results.json',ROOT/'results/merged-game1/development-score-tables.json',
            ROOT/'results/failure-analysis/results.json',ROOT/'results/human-challenge/results.json',
            ROOT/'results/game1-machine-checkpoint-v1.json',ROOT/'results/game1-merged-checkpoint-v2.json']
    signature={'configuration':cfg,'configuration_sha256':sha(args.config),'source_hashes':{str(p.relative_to(ROOT)):sha(p)for p in sources},
               'input_hashes':{str(p.relative_to(ROOT)):sha(p)for p in inputs},
               'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform()}}
    if out.exists():
        if not args.resume:raise SystemExit('Output exists. Use --resume or a new folder.')
        if read(out/'run-inputs.json')!=signature:raise SystemExit('Cannot resume changed sources or inputs.')
        if (out/'manifest.json').exists():print('Run complete. Use the independent verifier.');return
    else:
        if args.resume:raise SystemExit('No existing run')
        out.mkdir(parents=True);(out/'samples').mkdir();save(out/'run-inputs.json',signature)
        for p in sources:
            target=out/'source-snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    merged,defs,diagnostic,human=map(read,inputs[:4])
    if (out/'frozen-candidates.json').exists():
        freeze=read(out/'frozen-candidates.json');candidates=freeze['candidates'];primary=freeze['primary'];definitions=freeze['definitions'];seed_pool=[tuple(x)for x in read(out/'arena-seed-pool.json')]
    else:
        tables,names,objectives,critical_keys=prepare_selection(merged,diagnostic,defs,cfg)
        save(out/'selection-tables.json',{'names':names,'models':{k:pack(t)for k,t in zip(names,tables)},'objectives':objectives,'critical_keys':critical_keys})
        print('Selecting the three declared rules over every legal action',flush=True)
        searched=search_frontiers(tables,objectives,cfg['enumeration_batch_size'])
        save(out/'selection-frontiers.json',{k:v for k,v in searched.items()if k!='security_twice'})
        np.savez_compressed(out/'selection-security.npz',security_twice=searched['security_twice'])
        candidates={};lookup={};primary={}
        def add(x,role):
            x=tuple(x)
            if x not in lookup:
                cid=f'c{len(candidates):03d}';lookup[x]=cid;candidates[cid]={'allocation':x,'roles':[]}
            cid=lookup[x]
            if role not in candidates[cid]['roles']:candidates[cid]['roles'].append(role)
            return cid
        for cid,c in merged['candidates'].items():add(c['allocation'],'merged_'+cid)
        for k,m in diagnostic['maps'].items():
            for label in ['unrestricted','security2']:
                for p in m[label]['segments']+m[label]['boundaries']:add(p['allocation'],'diagnostic_'+k+'_'+label)
        for name,frontier in searched['frontiers'].items():
            for c,p in frontier.items():add(p['allocation'],name+'_security'+str(c))
        for name,p in merged['primary'].items():
            cid=add(p['allocation'],'primary_'+name);primary[name]={'allocation':tuple(p['allocation']),'candidate_id':cid,'origin':'merged frozen primary'}
        x=merged['candidates'][merged['label_ids']['M_known_model_max_min']]['allocation'];cid=add(x,'primary_previous_machine_max_min')
        primary['previous_machine_max_min']={'allocation':tuple(x),'candidate_id':cid,'origin':'pre-human baseline'}
        for o in objectives:
            chosen=nominal_winner(searched['frontiers'][o.name],Fraction(2));cid=add(chosen['allocation'],'primary_'+o.name)
            primary[o.name]={'allocation':chosen['allocation'],'candidate_id':cid,'origin':'new known-evidence rule','objective':o.name,
                             'selection_value':chosen['value'],'optimum_count':chosen['optimum_count'],'security_floor':2}
        actions=[tuple(x)for x in human['pool_actions']];rng=np.random.default_rng(cfg['seed']+7000)
        for j in range(cfg['seed_pool_extra']//2):
            bars=np.sort(rng.choice(104,4,replace=False));actions.append(tuple(map(int,np.diff(np.r_[-1,bars,104])-1)))
        extra=final_rows(final_definitions('challenge')[1],cfg['seed_pool_extra']//2,cfg['seed']+7001)
        actions.extend(tuple(map(int,x))for x in extra);seed_pool=list(dict.fromkeys(actions));save(out/'arena-seed-pool.json',seed_pool)
        definitions={phase:[{**d,'seed':cfg['seed']+10000*p+j}for j,d in enumerate(final_definitions(phase))]for p,phase in enumerate(cfg['phases'])}
        freeze={'created_unix':time.time(),'candidates':candidates,'primary':primary,'definitions':definitions,
                'selection_tables_sha256':sha(out/'selection-tables.json'),'selection_frontiers_sha256':sha(out/'selection-frontiers.json'),
                'seed_pool_sha256':sha(out/'arena-seed-pool.json'),'run_inputs_sha256':sha(out/'run-inputs.json')}
        save(out/'frozen-candidates.json',freeze)
    freeze_sha=sha(out/'frozen-candidates.json')
    if not(out/'evaluation-start.json').exists():save(out/'evaluation-start.json',{'started_unix':time.time(),'freeze_sha256':freeze_sha})
    tables={};models={};learning={};families={}
    def add_model(key,t,phase,kind,meta=None):
        br=best_response(t);tables[key]=t;models[key]={**(meta or {}),'phase':phase,'kind':kind,**pack(t),'best_response':br}
    def mixture(key,keys,weights,phase,kind):
        t=mix_tables([tables[k]for k in keys],weights);add_model(key,t,phase,kind,{'component_keys':keys,'component_weights':weights});return key
    for phase in cfg['phases']:
        families[phase]={};static_by_family={}
        for definition in definitions[phase]:
            key=definition['key'];path=out/'samples'/(key+'.npz')
            if path.exists():
                with np.load(path)as a:rows=a['allocations']
            else:rows=final_rows(definition,cfg['evaluation_sample_size'],definition['seed']);np.savez_compressed(path,allocations=rows.astype(np.uint8))
            add_model(key,rows_to_table(rows,key),phase,'fresh_static',{'definition':definition,'sample_file':str(path.relative_to(out)),
               'sample_size':len(rows),**sampled_metadata(rows,definition['seed'])})
            static_by_family.setdefault(definition['family'],[]).append(key)
            print('Fresh population: '+key,flush=True)
        for group,keys in static_by_family.items():families[phase][group]=mixture(phase+'_family_'+group,keys,[1]*len(keys),phase,'family_mean')
        static_family_keys=list(families[phase].values())
        balanced=mixture(phase+'_mix_balanced',static_family_keys,[1]*len(static_family_keys),phase,'static_mixture')
        coalition_tables={c:[d['key']for d in definitions[phase]if ''.join(str(i+1)for i in d['coalition'])==c and d['law']in ['soft','heaped','middle5','contrarian']]for c in ['35','45','125','134','234']}
        component_keys=[k for c in coalition_tables.values()for k in c]
        for name,weights in [('favorites',[Fraction(35,100),Fraction(1,10),Fraction(1,10),Fraction(35,100),Fraction(1,10)]),
                ('five_priority',[Fraction(1,4),Fraction(1,4),Fraction(1,4),Fraction(1,8),Fraction(1,8)]),
                ('five_avoiding',[Fraction(1,10),Fraction(1,10),Fraction(1,10),Fraction(35,100),Fraction(35,100)])]:
            mixture(phase+'_mix_'+name,component_keys,[w/4 for w in weights for _ in range(4)],phase,'static_mixture')
        for group in ['heaped','middle5']:mixture(phase+'_mix_'+group,[families[phase][group],balanced],[3,1],phase,'static_mixture')
        for windex,win_share in enumerate(cfg['win_shares']):
            for rindex,rule in enumerate(cfg['rules']):
                seed=cfg['seed']+30000+10000*cfg['phases'].index(phase)+100*windex+rindex
                rng=np.random.default_rng(seed);initial=rng.dirichlet(np.full(len(seed_pool),2 if phase=='challenge'else .08))
                mutation=.01 if phase=='challenge'else 0
                name=f'{phase}_w{win_share}_{rule}';folder=out/'learning'/name
                path=enhanced_path(folder,seed_pool,initial,win_share,rule,mutation,cfg,seed+1000)
                np.savez_compressed(folder/'arrays.npz',initial=initial,point=path['point'],win=path['win'],final=path['final'],average=path['average'])
                record={'phase':phase,'win_share':win_share,'rule':rule,'mutation':mutation,'initial_seed':seed,'path_seed':seed+1000,
                        'actions':path['actions'],'trace':path['trace'],'events':path['events'],'input_digest':path['input_digest']}
                for snapshot in ['final','average']:
                    pop=population_from_shares(path['actions'],path[snapshot],cfg['quantization_mass']);certificate=security_of(pop)
                    record[snapshot]={'population':pop.to_dict(),'certificate':certificate}
                    add_model(name+'_'+snapshot,pop.table(),phase,'fresh_learning',{'path':name,'snapshot':snapshot})
                save(folder/'result.json',record);learning[name]=record;print('Fresh arena complete: '+name,flush=True)
        for snapshot in ['final','average']:
            keys=[name+'_'+snapshot for name,p in learning.items()if p['phase']==phase]
            family=mixture(phase+'_family_arena_'+snapshot,keys,[1]*len(keys),phase,'family_mean');families[phase]['arena_'+snapshot]=family
            for raw in cfg['learner_mixture_shares']:
                share=Fraction(raw);mixture(phase+'_mix_learner_'+snapshot+'_'+raw,[balanced,family],[1-share,share],phase,'learner_mixture')
    print('Scoring every frozen candidate on all fresh and derived populations',flush=True)
    scores,regrets,summary=evaluate(candidates,tables,models,families)
    security={cid:dict(zip(['counter','guarantee'],exact_counter(c['allocation'])))for cid,c in candidates.items()}
    count=sum(len(v)for v in definitions.values());bound=15*math.sqrt(math.log(10*count/cfg['sampling_alpha'])/(2*cfg['evaluation_sample_size']))
    result={'configuration':cfg,'candidates':candidates,'primary':primary,'models':models,'families':families,'learning':learning,'security':security,
            'scores':scores,'regrets':regrets,'summary':summary,'sampling_bound':bound,'freeze_sha256':freeze_sha,
            'scope':'Frozen fresh constructed challenges, not applicant-frequency inference; final selection deferred'}
    save(out/'results.json',result);figures(out,result);report(out,result)
    write_csv(out/'primary-summary.csv',['rule','allocation','phase','family_mean','family_minimum','individual_minimum','individual_worst_regret','pure_security'],
      [[name,str(tuple(p['allocation'])),phase,*[float(summary[p['candidate_id']][phase][k])for k in ['family_balanced_mean','family_minimum','individual_minimum','individual_worst_regret']],float(security[p['candidate_id']]['guarantee'])]for name,p in primary.items()for phase in cfg['phases']])
    for p in sources:assert sha(p)==signature['source_hashes'][str(p.relative_to(ROOT))]
    save(out/'manifest.json',{'signature':signature,'freeze_sha256':freeze_sha,'completed_unix':time.time(),'elapsed_seconds':time.perf_counter()-clock,
          'artifacts':{str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()and p.name!='manifest.json'}})
    print('Final challenge complete. Run independent verification.',flush=True)


if __name__=='__main__':main()
