import { check, fail } from 'k6';
import sse from "k6/x/sse"
import { scenario } from 'k6/execution';
import { Trend, Counter } from 'k6/metrics';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.2/index.js';

const host = __ENV.HOST;
const apiKey = __ENV.HF_TOKEN;
const filePath = __ENV.FILE_PATH || "k6-summary.json";

const endToEndLatency = new Trend('end_to_end_latency', true);
const requestThroughput = new Counter('request_throughput');
const tokenThroughput = new Counter('tokens_throughput');
const timeToFirstToken = new Trend('time_to_first_token', true);
const interTokenLatency = new Trend('inter_token_latency', true); // is microseconds
const tokensReceived = new Trend('tokens_received');

const max_new_tokens = 200;
const dataset = JSON.parse(open("small.json"))

export function handleSummary(data) {
    return {
        stdout: textSummary(data, { indent: " ", enableColors: true }),
        [filePath]: JSON.stringify(data), //the default data object
    };
}


export function generate_payload(example, max_new_tokens) {
    const input = example[0]["value"]
    return {
        "messages": [{ "role": "user", "content": input }],
        "model": "tgi",
        "temperature": 0,
        "max_tokens": max_new_tokens,
        "stream": true
    };
}

export const options = get_options();

function get_options() {
    return {
        scenarios: {
            load_test: {
                executor: 'constant-arrival-rate',
                duration: '10s',
                preAllocatedVUs: 25,
                rate: 1,
                timeUnit: '1s',
            },
        },
    };
}

export default function run() {
    const headers = {
        Accept: "application/json",
        Authorization: "Bearer " + apiKey,
        "Content-Type": "application/json",
    };
    const query = dataset[scenario.iterationInTest % dataset.length];
    const payload = JSON.stringify(generate_payload(query, max_new_tokens));
    const url = `${host}/v1/chat/completions`;
    const params = {
        method: 'POST',
        body: payload,
        headers
    }

    const startTime = Date.now();
    let firstTokenTime = null;
    let lastTokenTime = null;
    let tokensCount = 0;
    let response = ""

    const res = sse.open(url, params, function (client) {
        client.on('event', function (event) {
            // console.log(event.data)
            if (parseInt(event.id) === 4) {
                client.close()
            }
            if (event.data.includes("[DONE]") || event.data === "") {
                return
            }
            try {
                const data = JSON.parse(event.data);
                if (!'choices' in data) {
                    fail('http_200')
                    return;
                }
                const content = data['choices'][0]['delta']['content']
                if (content !== undefined) {
                    response += data['choices'][0]['delta']['content']
                    tokensCount += 1;
                }

                // Measure time to first token
                if (!firstTokenTime) {
                    firstTokenTime = Date.now();
                    timeToFirstToken.add(firstTokenTime - startTime);
                }

                // Measure inter-token latency
                const currentTime = Date.now();
                if (lastTokenTime) {
                    interTokenLatency.add((currentTime - lastTokenTime) * 1000.);
                }
                lastTokenTime = currentTime;

                if ('finish_reason' in data['choices'][0]) {
                    if (data['choices'][0]['finish_reason'] != null) {
                        const endTime = Date.now();
                        const deltaMs = endTime - startTime;
                        endToEndLatency.add(deltaMs)
                        requestThroughput.add(1);
                        tokenThroughput.add(tokensCount);
                        tokensReceived.add(tokensCount);
                    }
                }
            } catch (e) {
                // catch any errors that occur during the event processing
                // increase the fail count of the 'http_200' check
                check(true, {
                    'http_200': (val) => false,
                })
                fail('http_200')
            }
        })

        client.on('error', function (e) {
            console.log('An unexpected error occurred: ', e.error())
        })
    })

    if (tokensCount === 0) {
        // something went wrong with generation
        fail('http_200')
    }

    if (res.status >= 400 && res.status < 500) {
        return;
    }

    check(res, {
        'http_200': (res) => res.status === 200,
    });

}