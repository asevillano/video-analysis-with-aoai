# Final video summary helpers.
#
# All logic to parse per-segment model outputs, accumulate items across segments,
# consolidate them with a single LLM call and render the final summary in the
# Streamlit UI lives here so the main application file stays focused on the
# segmentation / analysis loop.
import json
import os


# Try to parse the model analysis output into a Python dict. Returns None if it cannot be parsed.
def _parse_analysis_json(analysis):
    if isinstance(analysis, dict):
        return analysis
    if not isinstance(analysis, str):
        return None
    text = analysis.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].startswith('```'):
            lines = lines[:-1]
        text = '\n'.join(lines).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


# Extract a flat list of "items" (incidents / events / elements) from one segment analysis,
# regardless of the prompt schema used (generic, riots, abandoned-objects, combined).
def _extract_summary_items(parsed, segment_path, segment_start):
    items = []
    if not isinstance(parsed, dict):
        return items

    seg_label = os.path.basename(segment_path) if segment_path else ''

    # Schema: COMBINED -> top-level "incidents": [ {...}, ... ]
    if isinstance(parsed.get('incidents'), list) and parsed['incidents']:
        for inc in parsed['incidents']:
            if not isinstance(inc, dict):
                continue
            poi = inc.get('person_of_interest') or {}
            items.append({
                'kind': 'incident',
                'type': inc.get('incident_type', 'INCIDENT'),
                'classification': inc.get('classification', ''),
                'confidence': inc.get('confidence', ''),
                'time_start': inc.get('time_first_seen', ''),
                'time_end': inc.get('time_last_seen', ''),
                'location': inc.get('location_in_scene', ''),
                'indicators': inc.get('behavioral_indicators', []) or [],
                'people': inc.get('estimated_people_involved', ''),
                'action': inc.get('recommended_action', ''),
                'reasoning': inc.get('reasoning', ''),
                'person': ', '.join(filter(None, [poi.get('description', ''), poi.get('clothing', '')])) if isinstance(poi, dict) else '',
                'segment': seg_label,
                'segment_start': segment_start,
            })
        return items

    # Schema: RIOTS single-incident
    if 'incident_window' in parsed:
        if parsed.get('incident_detected'):
            window = parsed.get('incident_window') or {}
            items.append({
                'kind': 'incident',
                'type': 'RIOT',
                'classification': parsed.get('classification', ''),
                'confidence': parsed.get('confidence', ''),
                'time_start': window.get('time_first_seen', '') if isinstance(window, dict) else '',
                'time_end': window.get('time_last_seen', '') if isinstance(window, dict) else '',
                'location': parsed.get('scene_context', ''),
                'indicators': parsed.get('behavioral_indicators', []) or [],
                'people': parsed.get('estimated_people_involved', ''),
                'action': parsed.get('recommended_action', ''),
                'reasoning': parsed.get('reasoning', ''),
                'person': '',
                'segment': seg_label,
                'segment_start': segment_start,
            })
        return items

    # Schema: ABANDONED OBJECTS single-incident
    if 'abandoned_object' in parsed or 'person_of_interest' in parsed:
        if parsed.get('incident_detected'):
            obj = parsed.get('abandoned_object') or {}
            poi = parsed.get('person_of_interest') or {}
            items.append({
                'kind': 'incident',
                'type': 'ABANDONED_OBJECT',
                'classification': parsed.get('classification', ''),
                'confidence': parsed.get('confidence', ''),
                'time_start': obj.get('time_first_seen_unattended', '') if isinstance(obj, dict) else '',
                'time_end': poi.get('time_left_scene', '') if isinstance(poi, dict) else '',
                'location': obj.get('location_in_scene', '') if isinstance(obj, dict) else '',
                'indicators': parsed.get('behavioral_indicators', []) or [],
                'people': '',
                'action': parsed.get('recommended_action', ''),
                'reasoning': parsed.get('reasoning', ''),
                'person': ', '.join(filter(None, [poi.get('description', ''), poi.get('clothing', '')])) if isinstance(poi, dict) else '',
                'segment': seg_label,
                'segment_start': segment_start,
            })
        return items

    # Schema: GENERIC -> description / objects / actions
    if any(k in parsed for k in ('description', 'objects', 'actions')):
        description = parsed.get('description', '')
        objects = parsed.get('objects', []) or []
        actions = parsed.get('actions', []) or []
        if description or objects or actions:
            items.append({
                'kind': 'generic',
                'type': 'SCENE',
                'description': description,
                'objects': objects if isinstance(objects, list) else [objects],
                'actions': actions if isinstance(actions, list) else [actions],
                'segment': seg_label,
                'segment_start': segment_start,
            })
    return items


# Format a coarse segment time range as mm:ss-mm:ss for the generic summary table.
def _format_segment_range(segment_start, segment_duration):
    try:
        s = int(segment_start or 0)
        e = int(s + (segment_duration or 0))
        return f"{s // 60:02}:{s % 60:02}-{e // 60:02}:{e % 60:02}"
    except Exception:
        return ''


# Call the model once with all per-segment analyses to produce ONE consolidated summary
# of the WHOLE video. Returns a tuple (result_dict, usage_dict) where usage has
# prompt_tokens / completion_tokens / total_tokens (zeros on error).
def consolidate_video_summary(segment_results, aoai_client, aoai_model_name, reasoning_effort='medium'):
    if not segment_results:
        return None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Build a compact textual payload with one block per segment
    blocks = []
    for idx, seg in enumerate(segment_results, start=1):
        start = seg.get('segment_start', 0)
        try:
            start = int(start)
        except Exception:
            start = 0
        header = f"=== Segment {idx} (starts at {start//60:02}:{start%60:02} of the original video, file: {seg.get('segment','')}) ==="
        blocks.append(header + "\n" + (seg.get('analysis_text') or ''))
    joined = "\n\n".join(blocks)

    system_prompt = (
        "You are a video analysis consolidator. You will receive the JSON analyses of "
        "consecutive segments of the SAME video, in chronological order. Produce ONE "
        "consolidated summary of the WHOLE video. Merge duplicates across segments and "
        "preserve the absolute timestamps (mm:ss:msec) reported in the per-segment JSONs "
        "whenever available. Respond with a SINGLE JSON object that exactly matches this "
        "schema and nothing else:\n"
        "{\n"
        '  "overall_summary": "",\n'
        '  "events": [ { "description": "", "time_start": "", "time_end": "" } ],\n'
        '  "incidents": [ { "type": "", "classification": "", "confidence": "", "time_start": "", "time_end": "", "location": "", "description": "" } ],\n'
        '  "elements": { "objects": [], "actions": [], "people": [] }\n'
        "}\n"
        "Rules: copy timestamps verbatim from the segment JSONs (do not invent them). If a "
        "segment did not report timestamps for an event, leave time_start/time_end empty. "
        "Keep lists deduplicated."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Per-segment analyses to consolidate:\n\n{joined}"},
    ]

    # Retry with progressively larger budgets: reasoning models can consume the entire
    # max_completion_tokens on internal reasoning and return an empty message content,
    # which makes json.loads fail with "Expecting value: line 1 column 1 (char 0)".
    budgets = [8192, 16384, 32768]
    last_error = None
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for budget in budgets:
        try:
            response = aoai_client.chat.completions.create(
                model=aoai_model_name,
                messages=messages,
                max_completion_tokens=budget,
                reasoning_effort=reasoning_effort,
                response_format={"type": "json_object"},
            )
            choice = response.choices[0] if response.choices else None
            content = (getattr(choice.message, 'content', None) if choice else None) or ''
            finish_reason = getattr(choice, 'finish_reason', '') if choice else ''
            usage_obj = getattr(response, 'usage', None)
            usage = {
                "prompt_tokens": int(getattr(usage_obj, 'prompt_tokens', 0) or 0),
                "completion_tokens": int(getattr(usage_obj, 'completion_tokens', 0) or 0),
                "total_tokens": int(getattr(usage_obj, 'total_tokens', 0) or 0),
            }
            text = content.strip()
            if not text:
                last_error = f"empty response from model (finish_reason={finish_reason}, budget={budget})"
                print(f'WARN consolidating summary: {last_error}; retrying with larger budget...')
                continue
            # Strip ```json fences if present
            if text.startswith('```'):
                lines = text.splitlines()
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].startswith('```'):
                    lines = lines[:-1]
                text = '\n'.join(lines).strip()
            try:
                return json.loads(text), usage
            except (json.JSONDecodeError, ValueError) as je:
                last_error = f"invalid JSON (finish_reason={finish_reason}, budget={budget}): {je}"
                print(f'WARN consolidating summary: {last_error}; retrying with larger budget...')
                continue
        except Exception as ex:
            last_error = str(ex)
            print(f'ERROR consolidating summary (budget={budget}): {ex}')
            continue

    return {'_error': last_error or 'unknown error'}, usage


# Render a final summary of the WHOLE video at the end of the analysis.
def render_final_summary(st, collected_items, segment_results, aoai_client, aoai_model_name,
                         reasoning_effort='medium', segment_duration_hint=0):
    st.markdown("---")
    st.header("📋 Video summary")

    # 1) Global incidents table (aggregated across all segments)
    incidents = [it for it in collected_items if it.get('kind') == 'incident']
    if incidents:
        st.subheader(f"🚨 Detected incidents ({len(incidents)})")
        rows = []
        for it in incidents:
            if it.get('time_start') and it.get('time_end') and it['time_start'] != it['time_end']:
                time_range = f"{it['time_start']} → {it['time_end']}"
            else:
                time_range = it.get('time_start') or it.get('time_end') or ''
            rows.append({
                'Type': it.get('type', ''),
                'Classification': it.get('classification', ''),
                'Confidence': it.get('confidence', ''),
                'Time': time_range,
                'Location': it.get('location', ''),
                'People': it.get('people', ''),
                'Recommended action': it.get('action', ''),
            })
        try:
            st.dataframe(rows, width='stretch', hide_index=True)
        except Exception:
            st.table(rows)

        with st.expander("Details (indicators and reasoning)", expanded=False):
            for idx, it in enumerate(incidents, start=1):
                st.markdown(f"**{idx}. {it.get('type','')} — {it.get('classification','')}**")
                if it.get('indicators'):
                    st.markdown("- Indicators: " + ", ".join(str(x) for x in it['indicators']))
                if it.get('person'):
                    st.markdown(f"- Person of interest: {it['person']}")
                if it.get('reasoning'):
                    st.markdown(f"- Reasoning: {it['reasoning']}")

    # 2) Consolidated summary of the WHOLE video (single LLM call over all segment outputs)
    if not segment_results:
        if not incidents:
            st.info("No events, incidents or elements were reported by the model.")
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    with st.spinner("Generating consolidated video summary..."):
        consolidated, summary_usage = consolidate_video_summary(
            segment_results, aoai_client, aoai_model_name, reasoning_effort=reasoning_effort
        )

    if not consolidated or consolidated.get('_error'):
        st.warning(f"Could not generate consolidated summary: {consolidated.get('_error','unknown error') if consolidated else 'no output'}")
        return summary_usage

    if consolidated.get('overall_summary'):
        st.subheader("🎬 Overall summary")
        st.markdown(consolidated['overall_summary'])

    events = consolidated.get('events') or []
    if events:
        st.subheader(f"📌 Events ({len(events)})")
        ev_rows = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            if ev.get('time_start') and ev.get('time_end') and ev['time_start'] != ev['time_end']:
                t = f"{ev['time_start']} → {ev['time_end']}"
            else:
                t = ev.get('time_start') or ev.get('time_end') or ''
            ev_rows.append({'Time': t, 'Description': ev.get('description', '')})
        try:
            st.dataframe(ev_rows, width='stretch', hide_index=True)
        except Exception:
            st.table(ev_rows)

    cons_incidents = consolidated.get('incidents') or []
    if cons_incidents and not incidents:
        # Only show this section when we did NOT already render the per-incident table above
        st.subheader(f"🚨 Incidents ({len(cons_incidents)})")
        try:
            st.dataframe(cons_incidents, width='stretch', hide_index=True)
        except Exception:
            st.table(cons_incidents)

    elements = consolidated.get('elements') or {}
    if isinstance(elements, dict) and (elements.get('objects') or elements.get('actions') or elements.get('people')):
        st.subheader("🧩 Elements identified")
        if elements.get('objects'):
            st.markdown("- **Objects:** " + ", ".join(str(x) for x in elements['objects']))
        if elements.get('actions'):
            st.markdown("- **Actions:** " + ", ".join(str(x) for x in elements['actions']))
        if elements.get('people'):
            st.markdown("- **People:** " + ", ".join(str(x) for x in elements['people']))

    return summary_usage
